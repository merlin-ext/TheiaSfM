// Author:: Raul

#include <Eigen/Core>
#include <vector>

#include "theia/matching/flann_feature_matcher.h"
#include "theia/matching/distance.h"
#include "theia/matching/feature_matcher.h"
#include "theia/matching/image_pair_match.h"

#include "gtest/gtest.h"

namespace theia {

using Eigen::VectorXf;

static const int kNumDescriptors = 10;
static const int kNumDescriptorDimensions = 10;

TEST(FlannFeatureMatcherTest, NoOptionsInCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1(kNumDescriptors);
  std::vector<VectorXf> descriptor2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    // Avoid a zero vector.
    descriptor1[i] = VectorXf::Constant(kNumDescriptorDimensions, 1);
    descriptor2[i] = VectorXf::Constant(kNumDescriptorDimensions, 1);
    descriptor1[i].normalize();
    descriptor2[i].normalize();
  }

  // Set options.
  FeatureMatcherOptions options;
  options.match_out_of_core = false;
  options.keypoints_and_descriptors_output_dir = "";
  options.min_num_feature_matches = 0;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = false;
  options.perform_geometric_verification = false;

  // Add features.
  std::vector<Keypoint> keypoints1(descriptor1.size());
  std::vector<Keypoint> keypoints2(descriptor2.size());
  FlannFeatureMatcher matcher(options);
  matcher.AddImage("1", keypoints1, descriptor1);
  matcher.AddImage("2", keypoints2, descriptor2);

  // Match features
  std::vector<ImagePairMatch> matches;
  matcher.MatchImages(&matches);

  // Check that the results are valid.
  EXPECT_EQ(matches[0].correspondences.size(), kNumDescriptors);
}

TEST(FlannFeatureMatcherTest, RatioTestInCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1(1);
  std::vector<VectorXf> descriptor2(2);
  descriptor1[0] = VectorXf::Constant(kNumDescriptorDimensions, 1).normalized();

  // Set the two descriptors to be very close to each other so that they do not
  // pass the ratio test.
  descriptor2[0] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[0](0) = 0.9;
  descriptor2[0].normalize();
  descriptor2[1] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[1](0) = 0.89;
  descriptor2[1].normalize();

  // Set options.
  FeatureMatcherOptions options;
  options.match_out_of_core = false;
  options.keypoints_and_descriptors_output_dir = "";
  options.min_num_feature_matches = 0;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = true;
  options.perform_geometric_verification = false;

  // Add features.
  std::vector<Keypoint> keypoints1(descriptor1.size());
  std::vector<Keypoint> keypoints2(descriptor2.size());
  FlannFeatureMatcher matcher(options);
  matcher.AddImage("1", keypoints1, descriptor1);
  matcher.AddImage("2", keypoints2, descriptor2);

  // Match features.
  std::vector<ImagePairMatch> matches;
  matcher.MatchImages(&matches);

  // Check that the results are valid.
  EXPECT_EQ(matches[0].correspondences.size(), 0);
}

TEST(FlannFeatureMatcherTest, SymmetricMatchesInCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1(2);
  std::vector<VectorXf> descriptor2(2);
  descriptor1[0] = VectorXf::Constant(kNumDescriptorDimensions, 1).normalized();
  descriptor1[1] = VectorXf::Constant(kNumDescriptorDimensions, 0);
  descriptor1[1](0) = 1.0;

  // Set the two descriptors to be closer to descriptor1[0] so that the
  // symmetric matching produces only 1 match.
  descriptor2[0] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[0](0) = 0;
  descriptor2[0].normalize();
  descriptor2[1] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[1](1) = 0;
  descriptor2[1](2) = 0;
  descriptor2[1].normalize();

  // Set options.
  FeatureMatcherOptions options;
  options.match_out_of_core = false;
  options.keypoints_and_descriptors_output_dir = "";
  options.min_num_feature_matches = 0;
  options.keep_only_symmetric_matches = true;
  options.use_lowes_ratio = false;
  options.perform_geometric_verification = false;

  // Add features.
  std::vector<Keypoint> keypoints1(descriptor1.size());
  std::vector<Keypoint> keypoints2(descriptor2.size());
  FlannFeatureMatcher matcher(options);
  matcher.AddImage("1", keypoints1, descriptor1);
  matcher.AddImage("2", keypoints2, descriptor2);

  // Match features.
  std::vector<ImagePairMatch> matches;
  matcher.MatchImages(&matches);

  // Check that the results are valid.
  EXPECT_EQ(matches[0].correspondences.size(), 1);
}

TEST(FlannFeatureMatcherTest, NoOptionsOutOfCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1(kNumDescriptors);
  std::vector<VectorXf> descriptor2(kNumDescriptors);
  for (int i = 0; i < kNumDescriptors; i++) {
    // Avoid a zero vector.
    descriptor1[i] = VectorXf::Constant(kNumDescriptorDimensions, 1);
    descriptor2[i] = VectorXf::Constant(kNumDescriptorDimensions, 1);
    descriptor1[i].normalize();
    descriptor2[i].normalize();
  }

  // Set options.
  FeatureMatcherOptions options;
  options.match_out_of_core = true;
  options.keypoints_and_descriptors_output_dir = GTEST_TESTING_OUTPUT_DIRECTORY;
  options.min_num_feature_matches = 0;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = false;
  options.perform_geometric_verification = false;

  // Add features.
  std::vector<Keypoint> keypoints1(descriptor1.size());
  std::vector<Keypoint> keypoints2(descriptor2.size());
  FlannFeatureMatcher matcher(options);
  matcher.AddImage("1", keypoints1, descriptor1);
  matcher.AddImage("2", keypoints2, descriptor2);

  // Match features
  std::vector<ImagePairMatch> matches;
  matcher.MatchImages(&matches);

  // Check that the results are valid.
  EXPECT_EQ(matches[0].correspondences.size(), kNumDescriptors);
}

TEST(FlannFeatureMatcherTest, RatioTestOutOfCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1(1);
  std::vector<VectorXf> descriptor2(2);
  descriptor1[0] = VectorXf::Constant(kNumDescriptorDimensions, 1).normalized();

  // Set the two descriptors to be very close to each other so that they do not
  // pass the ratio test.
  descriptor2[0] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[0](0) = 0.9;
  descriptor2[0].normalize();
  descriptor2[1] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[1](0) = 0.89;
  descriptor2[1].normalize();

  // Set options.
  FeatureMatcherOptions options;
  options.match_out_of_core = true;
  options.keypoints_and_descriptors_output_dir = GTEST_TESTING_OUTPUT_DIRECTORY;
  options.min_num_feature_matches = 0;
  options.keep_only_symmetric_matches = false;
  options.use_lowes_ratio = true;
  options.perform_geometric_verification = false;

  // Add features.
  std::vector<Keypoint> keypoints1(descriptor1.size());
  std::vector<Keypoint> keypoints2(descriptor2.size());
  FlannFeatureMatcher matcher(options);
  matcher.AddImage("1", keypoints1, descriptor1);
  matcher.AddImage("2", keypoints2, descriptor2);

  // Match features.
  std::vector<ImagePairMatch> matches;
  matcher.MatchImages(&matches);

  // Check that the results are valid.
  EXPECT_EQ(matches[0].correspondences.size(), 0);
}

TEST(FlannFeatureMatcherTest, SymmetricMatchesOutOfCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1(2);
  std::vector<VectorXf> descriptor2(2);
  descriptor1[0] = VectorXf::Constant(kNumDescriptorDimensions, 1).normalized();
  descriptor1[1] = VectorXf::Constant(kNumDescriptorDimensions, 0);
  descriptor1[1](0) = 1.0;

  // Set the two descriptors to be closer to descriptor1[0] so that the
  // symmetric matching produces only 1 match.
  descriptor2[0] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[0](0) = 0;
  descriptor2[0].normalize();
  descriptor2[1] = VectorXf::Constant(kNumDescriptorDimensions, 1);
  descriptor2[1](1) = 0;
  descriptor2[1](2) = 0;
  descriptor2[1].normalize();

  // Set options.
  FeatureMatcherOptions options;
  options.match_out_of_core = true;
  options.keypoints_and_descriptors_output_dir = GTEST_TESTING_OUTPUT_DIRECTORY;
  options.min_num_feature_matches = 0;
  options.keep_only_symmetric_matches = true;
  options.use_lowes_ratio = false;
  options.perform_geometric_verification = false;

  // Add features.
  std::vector<Keypoint> keypoints1(descriptor1.size());
  std::vector<Keypoint> keypoints2(descriptor2.size());
  FlannFeatureMatcher matcher(options);
  matcher.AddImage("1", keypoints1, descriptor1);
  matcher.AddImage("2", keypoints2, descriptor2);

  // Match features.
  std::vector<ImagePairMatch> matches;
  matcher.MatchImages(&matches);

  // Check that the results are valid.
  EXPECT_EQ(matches[0].correspondences.size(), 1);
}

}  // namespace theia
