import glob
import numpy as np
import os
import shutil
from src.logger.logger import _logger


def to_filelist(args, mode="train"):
    if mode == "train":
        flist = args.data_train
    elif mode == "val":
        flist = args.data_val
    else:
        raise NotImplementedError("Invalid mode %s" % mode)
    print(flist)
    # keyword-based: 'a:/path/to/a b:/path/to/b'
    file_dict = {}
    for f in flist:
        if ":" in f:
            name, fp = f.split(":")
        else:
            name, fp = "_", f
        files = glob.glob(fp)
        if name in file_dict:
            file_dict[name] += files
        else:
            file_dict[name] = files
    # sort files
    for name, files in file_dict.items():
        file_dict[name] = sorted(files)

    if args.local_rank is not None:
        if mode == "train":
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[args.local_rank :: local_world_size]
                assert len(new_files) > 0
                np.random.shuffle(new_files)
                new_file_dict[name] = new_files
            file_dict = new_file_dict

    if args.copy_inputs:
        import tempfile

        tmpdir = tempfile.mkdtemp()
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        new_file_dict = {name: [] for name in file_dict}
        for name, files in file_dict.items():
            for src in files:
                dest = os.path.join(tmpdir, src.lstrip("/"))
                if not os.path.exists(os.path.dirname(dest)):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                _logger.info("Copied file %s to %s" % (src, dest))
                new_file_dict[name].append(dest)
            if len(files) != len(new_file_dict[name]):
                _logger.error(
                    "Only %d/%d files copied for %s file group %s",
                    len(new_file_dict[name]),
                    len(files),
                    mode,
                    name,
                )
        file_dict = new_file_dict

    filelist = sum(file_dict.values(), [])
    assert len(filelist) == len(set(filelist))
    return file_dict, filelist
