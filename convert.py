import rawpy
import imageio.v3 as iio
import glob, os

in_dir = "C:\\Users\\SSAFY\\Desktop\\Photos"
out_dir = "C:\\Users\\SSAFY\\Desktop\\Photos"
os.makedirs(out_dir, exist_ok=True)

for f in glob.glob(os.path.join(in_dir, "*.RW2")):
    name = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(out_dir, name + ".jpg")

    with rawpy.imread(f) as raw:
        rgb = raw.postprocess(use_auto_wb=True, no_auto_bright=True, output_bps=8)
    iio.imwrite(out, rgb, extension=".jpg")

    print("✔", out)