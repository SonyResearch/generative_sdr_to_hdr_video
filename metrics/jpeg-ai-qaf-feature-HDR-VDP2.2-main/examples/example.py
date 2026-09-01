import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.getcwd(), os.path.pardir))

from metrics import MetricsProcessor  # noqa: E402


def new_layer_flops_counter_hook(new_layer_module, input, output):
    # Took from https://github.com/sovrasov/flops-counter.pytorch/blob/5f2a45f8ff117ce5ad34a466270f4774edd73379/ptflops/pytorch_ops.py#L53
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])
    kernel_size = [1, 1]

    kernel_dims = list(kernel_size)
    in_channels = 3  # new_layer_module.c.in_channels
    out_channels = 3  # new_layer_module.c.out_channels
    groups = 3  # new_layer_module.c.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    # if new_layer_module.c.bias is not None:

    #    bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    new_layer_module.__flops__ += int(overall_flops)


class NewLayer(nn.Module):
    def __init__(self):
        super(NewLayer, self).__init__()
        self.weights = nn.Parameter(torch.ones((3, 1, 1, 1)))
        self.weights.fill_(0.5)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weights, groups=3)


custom_modules = {NewLayer: new_layer_flops_counter_hook}


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.m = NewLayer()

    def forward(self, x):
        return self.m(x)


def simple_test_of_model(model, shape):
    # Simple example from github: https://github.com/sovrasov/flops-counter.pytorch
    from ptflops import get_model_complexity_info
    from ptflops.flops_counter import flops_to_string
    macs, params = get_model_complexity_info(
        model,
        tuple(shape),
        as_strings=False,
        print_per_layer_stat=True,
        verbose=True,
        custom_modules_hooks=custom_modules)
    print('==================')
    print('Simple model Flops: {0}, i.e {1} / pxl'.format(
        flops_to_string(macs, units=None),
        flops_to_string(macs / (shape[-1] * shape[-2]), units='Mac')))
    print('==================')

    return macs


def recommended_way_to_test_model(model, ori_image):
    import ptflops
    from ptflops.flops_counter import flops_to_string

    ptflops.flops_counter.CUSTOM_MODULES_MAPPING = custom_modules

    # Recomended design for decoder
    model = MetricsProcessor.init_ptflops_calc(model)

    # Should be 65536 * 100 operations
    rec = model(ori_image)

    flops_count, flops_per_pixel = MetricsProcessor.finish_ptflops_calc(
        model, ori_image.shape[-2:])
    print('==================')
    print('Recommended model: Flops: {0}, i.e {1} / pxl'.format(
        flops_to_string(flops_count, units=None),
        flops_to_string(flops_per_pixel, units='Mac')))
    print('==================')

    return rec, flops_count, flops_per_pixel


def store_tensor(fn, arr):
    from PIL import Image
    tmp = arr.permute(0, 2, 3, 1).squeeze(0)
    im = Image.fromarray(tmp.detach().mul(255).cpu().numpy().astype(np.uint8),
                         'RGB')
    im.save(fn)


def compare_files(fn1, fn2, sep):
    with open(fn1, 'r') as f1, open(fn2, 'r') as f2:
        s1 = f1.read()
        s2 = f2.read()
    s1s = s1.split(sep)
    s2s = s2.split(sep)
    return s1s[:15] == s2s[:15]


if __name__ == '__main__':

    # Initialize object for metrics calculation
    proc = MetricsProcessor()

    # assert False, 'Stop！'
    ap = argparse.ArgumentParser()
    # Add command-line arguments from metrics calculation class (if it is needed)
    ap.add_argument('--csv',
                    default=False,
                    action='store_true',
                    help='Output file in CSV format')
    proc.add_arguments(ap)
    # Parse teh arguments
    proc.parse_arguments(ap)
    args = ap.parse_args()

    summary_fn = 'summary.csv' if args.csv else 'summary.txt'
    summary_ref_fn = 'summary_ref.csv' if args.csv else 'summary_ref.txt'

    with torch.no_grad():
        ori_image = torch.arange(
            0, 256.0, dtype=torch.float).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        ori_image = ori_image.repeat([1, 3, 256, 1])
        ori_image = ori_image / 256.0

        model = TestModule()

        # For MACs calculation

        # 1) Simple test from authors of ptflops. We use it just as reference for checking recomended way of calculation.
        macs_count_1 = simple_test_of_model(model, ori_image.shape[1:])

        # 2) Recommended way of complexity calculation with "the real data".
        start_time = datetime.now()
        rec, macs_count_2, macs_per_pixel = recommended_way_to_test_model(
            model, ori_image)
        total_time = datetime.now() - start_time
        total_sec = total_time.total_seconds()

        # Check that tests 1 and 2 provides exactly the same results.
        assert macs_count_1 == macs_count_2, 'Flops are not the same'
        print('Test 1 and 2 matched')

        # Metrics calculation part

        # Store data to PNG file and bitstream
        ori_fn = 'ori.png'
        rec_fn = 'rec.png'
        bs_fn = 'data.bit'
        store_tensor(ori_fn, ori_image)
        store_tensor(rec_fn, rec)
        # Fill bitstream file by zeros. Total bits is 80, i.e. bpp is equal to 80 / (256*256) = 0.001220703125
        with open(bs_fn, 'wb') as f:
            a = np.zeros([10], dtype=np.uint8)
            a.tofile(f)
        proc.store_complexity_info(rec_fn,
                                   kmac=macs_per_pixel,
                                   decGPU=total_sec)

        # Initialization of the output file
        proc.init_summary_file(summary_fn, 'csv' if args.csv else 'txt')
        # Metrics calculation based on original and reconstructed file
        metrics = proc.process_image_files(ori_fn, rec_fn)
        # BPP calculation based on a size of the bitstream
        bpp = proc.bpp_calc(rec_fn, rec.shape)
        # Add complexity to output
        metrics += [macs_count_2]
        # Get additional information about complexity
        complexity_info = proc.load_complexity_info(rec_fn)
        # Write all metrics to the output file. `prefix_data` is only needed in CSV mode
        proc.write_data('test',
                        bpp,
                        metrics,
                        prefix_data=[ori_fn, 'CODEC_NAME'],
                        postfix_list=complexity_info)
        # Close the output file
        proc.close_file()

        # Compare output text file with reference
        print('====== Compare metrics =======')
        ans = compare_files(summary_ref_fn, summary_fn, proc.sep)
        print('Metrics matched' if ans else
              'There are mismatches between reference and test metrics')
