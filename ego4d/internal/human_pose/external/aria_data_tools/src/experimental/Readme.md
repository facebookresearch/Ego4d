# Experimental features

## VRS Copy Image mutation
`vrs_mutable` is a sample demonstrating how to create a VRS file copy with on the fly custom image modification.

### How the VRS image mutation is implemented

The feature is implemented by using the `RecordFilterCopier` concept along a `vrs copy` operation.

By adding a `ImageMutationFilter` abstraction of `RecordFilterCopier` with a custom `UserDefinedImageMutator`, the user can modify on the fly the PixelFrame image read from a given VRS and export it to a new VRS file.
FYI:
- Image size and bit depth must remains the same.
- `ImageMutationFilter::shouldCopyVerbatim` implements the logic to apply the functor only on image stream
- `ImageMutationFilter::filterImage` implements the JPG buffer codec logic and allow you to access the uncompressed PixelFrame for mutation with your functor

### How to write your how custom image mutation?
- See the provided examples `VerticalImageFlipMutator`, `NullifyModuloTwoTimestamp` or extend the `VrsExportLoader`.


```
struct MyCustomImageMutator : public vrs::utils::UserDefinedImageMutator {
  bool operator()(double timestamp, const vrs::StreamId& streamId, vrs::utils::PixelFrame* frame)
      override {
    if (!frame) { // If frame is invalid, Do nothing.
      return false;
    }
    // If frame is valid:
    // - apply your image processing function
    // - or load an image from disk to replace the image buffer
    // -> Note the image size (Width, Height, Stride) must be left unchanged

    // Image is defined by its PixelFrame:
    // frame->getWidth()
    // frame->getHeight()
    //
    // frame->wdata() - Pixel image buffer start (of size frame->getStride()*frame->getHeight() )
    //
    // You could iterate the image from top to bottom as following:
    //
    // const size_t lineLength = frame->getStride();
    // uint32_t top = 0;
    // uint32_t bottom = frame->getHeight() - 1;
    // while (top < bottom) {
    //   uint8_t* topPixels = frame->wdata() + top * frame->getStride();
    //   top++;
    // }

    return true;
  }
};
```

### How to run the code:
`vrs_mutable -- <VRS_IN> <VRS_OUT>`

### Illustration of a possible End2End demo (VRS export, local image mutation and VRS Copy mutation)

1. Export VRS Frames
```
$ vrs extract-images <VRS> --to /tmp/data
```

2. Modify your image frame (i.e blur some objects)

3. Use VRS Copy with ImageMutation to upload modified frames by using the `VrsExportLoader` Mutator
- Todo left as tutorial for the user (load the image frame and replace the pixel in PixelFrame)
