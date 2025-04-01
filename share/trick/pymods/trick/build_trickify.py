from pathlib import Path
import os

def_header_ext = [".h", "hh", "hpp", "H", "hxx", "h++"]
def_src_ext = ["cpp", "c"]

def find_files_by_extension(loc, ext):
    file_list = []
    for root, dirs, files in os.walk(loc):
        for f in files:
            if f.endswith(ext):
                file_list.append(os.path.join(root, f))
    return file_list

def build_S_source():
    loc = ""
    if "TRICKIFY_HEADER" in os.environ:
        loc = os.getenv("TRICKIFY_HEADER")
        dirs = loc.split()

    s_source = open("S_source.hh", 'w')

    for path in dirs:
        for ext in def_header_ext:
            files = find_files_by_extension(path, ext)
            for i in range(len(files)):
                s_source.write('#include "' + str(files[i]) + '"\n')

def build_obj_list():
    loc = ""
    if "TRICKIFY_SOURCE" in os.environ:
        loc = os.getenv("TRICKIFY_SOURCE")
        dirs = loc.split()

    obj_list = open("trickify_obj_list", 'w')

    for path in dirs:
        files = find_files_by_extension(path, "o")
        for i in range(len(files)):
            obj_list.write(str(files[i]) + '\n')

def build_src_list():
    loc = ""
    if "TRICKIFY_SOURCE" in os.environ:
        loc = os.getenv("TRICKIFY_SOURCE")
        dirs = loc.split()

    src_list = open("trickify_src_list", 'w')

    for path in dirs:
        for ext in def_src_ext:
            files = find_files_by_extension(path, ext)
            for i in range(len(files)):
                src_list.write(str(files[i]) + '\n')

def build_S_overrides_trickify(ldflags, ext_lib_dirs, python_path, swig_flags):
    s_overrides = open("S_overrides_trickify.mk", 'w')

    s_overrides.write("TRICK_LDFLAGS += " + ldflags + '\n')
    s_overrides.write("TRICK_EXT_LIB_DIRS += " + ext_lib_dirs + '\n')
    s_overrides.write("TRICK_PYTHON_PATH += " + python_path + '\n')
    s_overrides.write("TRICK_SWIG_FLAGS += -I" + swig_flags + '\n')
