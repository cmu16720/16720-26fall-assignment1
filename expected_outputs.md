# 1. PA1 Expected Outputs and Sanity Checks

These checks help diagnose coordinate, border, and ordering mistakes without revealing the required implementation.

## 2. Detector checks

- Gradient, second-moment, and response arrays preserve the input image shape.
- Centered differences are zero only on the derivative's two directional borders; they do not erase the full image border.
- A constant image has zero gradients and no retained positive Harris points.
- The supplied square places the strongest retained points at its four geometric corners, in deterministic score/coordinate order.
- Increasing the NMS radius should not increase the number of returned points.

## 3. Descriptor checks

- With the released settings, every descriptor has 128 entries.
- Non-flat descriptors have L2 norm just below or numerically equal to one; flat patches remain exactly zero.
- A positive global contrast scale largely cancels under normalization, while blur, rotation, and scale change are not intrinsically removed by this descriptor.
- A horizontal positive gradient votes into the released zero-angle bin in every nonempty cell.

## 4. Matching checks

- Empty inputs or fewer than two valid destination descriptors produce correctly shaped empty outputs.
- Returned match rows and confidence values have equal length and are ordered by decreasing confidence with deterministic index ties.
- On the supplied easy pair, a correct implementation should find enough geometrically consistent correspondences for the supplied affine estimator; exact counts can change only when documented parameters change.

## 5. Video checks

- The first frame supplies the reference ROI; later estimates map current-frame points back to the reference frame.
- A failed estimate reuses or marks a frame according to the documented failure policy rather than crashing or silently applying an invalid warp.
- The final labeled comparison is 10--20 seconds, H.264/yuv420p, at most 720p/30 fps, and no larger than 50 MB.
