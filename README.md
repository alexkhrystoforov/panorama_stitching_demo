Stitcher pipeline:
 1. Detect kp and compute desc using SIFT
 2. Match (BF matcher) the descriptors between the two images.
 3. Use ransak to estimate a homography matrix using our matched feature vectors
 4. Apply a warping transformation using the homography matrix 
