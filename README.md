This is a simple command line tool, and rudimentary library, for converting Nanoscope AFM files to images.  It's primarily intended for our own research use, and may not work well with other files.

It can be installed with

```
pip install -U git+https://github.com/cgevans/convert-nanoscope
```

The command line tool has basic help with `convert-nanoscope --help`.

# Changelog

## 0.1.0 (2023-11-05)

- Don't crash when file conversions fail.
- Add nanoscope-like colormap.