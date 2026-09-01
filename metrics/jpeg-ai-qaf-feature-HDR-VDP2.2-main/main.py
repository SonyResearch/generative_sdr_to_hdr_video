import argparse
import os
import re
from metrics import MetricsProcessor, DataClass


def find_files(lst, prefix, ext):
    ans = []
    for fn in lst:
        if fn.startswith(prefix) and fn.endswith(ext):
            ans.append(fn)
    return ans


def main():
    metrics = MetricsProcessor()
    ap = argparse.ArgumentParser()
    ap.add_argument('orig', default=os.getcwd(), help="Path original YUV/PNG file(s)")
    ap.add_argument('base_dir', default=None, help="Path to the base directory with results of different codecs")
    ap.add_argument('codec', default=None, help="Name of the codec")
    ap.add_argument('-b', default=None, help="Path to bitstreams")
    ap.add_argument('-lst', default=None, help="Path list of YUV/PNG file(s) to be processed")
    ap.add_argument('-r', default=None, help="Path reconstructed YUV/PNG file(s)")
    ap.add_argument('-s', default=None, help="Path output file with statistics")
    ap.add_argument('--format', default="yuv", choices=["yuv", "png"], help="Format of input image files")
    ap.add_argument('--bin-ext', default="bits", help="Extension of bin files")
    ap.add_argument('--no-bpp', default=False, action="store_true", help="Don't calculate bpp")
    ap.add_argument('-v', default=False, action="store_true", help="Verbose mode")
    ap.add_argument('--rates', nargs="+", type=int, default=[6, 12, 25, 50, 75], help="List of the rates to be checked")
    ap.add_argument('--csv', default=False, action="store_true", help="Output file in CSV format")
    metrics.add_arguments(ap)

    args = ap.parse_args()
    metrics.parse_arguments(ap)

    if args.r is None:
        args.r = os.path.join(args.base_dir, args.codec, "rec")
    if args.b is None:
        args.b = os.path.join(args.base_dir, args.codec, "bit")
    if args.s is None:
        if args.csv:
            args.s = f"{args.codec}_summary.csv"
        else:
            args.s = f"{args.codec}_summary.txt"

    flst = []
    if args.lst is None or (not os.path.exists(args.lst)):
        flst = [x for x in os.listdir(args.orig) if os.path.splitext(x.lower())[1][1:] == args.format.lower()]
        flst = sorted(flst)
        print(args.orig, os.listdir(args.orig))
        print("flst:", flst)
    else:
        with open(args.lst, "r") as iflst:
            for ifn in iflst:
                flst.append(ifn)

    lst_r = os.listdir(args.r)
    r = re.compile("(?P<name>.*)_(\d+)x(\d+)_(.+)_(?P<qp>\d+)")

    metrics.init_summary_file(args.s, "csv" if args.csv else "txt")

    for ifn in flst:
        # Iterate over all original files
        ifn = ifn.strip()
        bn, ext = os.path.splitext(ifn)
        rec_fns = find_files(lst_r, f"{args.codec}_{bn}", ext)
        # TODO: round data here
        if args.v:
            print(f"Load original file {ifn}")
        data_o, target_bd = DataClass().load_image(os.path.join(args.orig, ifn), def_bits=metrics.internal_bits,
                                                   color_conv=metrics.color_conv)
        for rate in args.rates:
            # Iterate over rates
            rec_fn = find_files(rec_fns, "", f"{rate:03d}{ext}")

            rec_fn = rec_fn[0] if len(rec_fn) > 0 else ""

            bn_r, ext_r = os.path.splitext(rec_fn)
            n = r.search(bn_r)
            if n is not None:
                bn_r = f"{n.group('name')}_{n.group('qp')}"
            bs_fn = os.path.join(args.b, f"{bn_r}.{args.bin_ext}")
            if args.v:
                print(f"Start processing {rec_fn}. ", end="")

            if len(rec_fn) == 0 or not os.path.exists(os.path.join(args.r, rec_fn)):
                if args.v:
                    print(f"No reconstructed file with bitrate {rate} for original {ifn}")
                metrics.write_data(f"No_rec_for_{ifn}_RATE{rate:03d}")
            else:
                data_r, _ = DataClass().load_image(os.path.join(args.r, rec_fn), def_bits=target_bd,
                                                   color_conv=metrics.color_conv)
                if args.no_bpp:
                    bpp = 0
                elif not os.path.exists(bs_fn):
                    if args.v:
                        print(f"No bitstream file {bs_fn}!")
                    bpp = -1
                else:
                    bpp = metrics.bpp_calc(bs_fn, data_r.shape)
                metrics_vals = metrics.process_images(data_o, data_r)
                print('metrics_vals', metrics_vals)
                if args.v:
                    print("Done.")
                complexity_data = metrics.load_complexity_info(os.path.join(args.r, rec_fn))

                metrics.write_data(rec_fn, bpp=bpp, metrics=metrics_vals, prefix_data=[ifn, args.codec],
                                   postfix_list=complexity_data)
    metrics.close_file()


if __name__ == "__main__":
    main()
