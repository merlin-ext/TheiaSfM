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

static const int kNumDescriptors = 5000;
static const int kNumDescriptorDimensions = 64;
std::shared_ptr<RandomNumberGenerator> rng = std::make_shared<RandomNumberGenerator>(55);

TEST(FlannFeatureMatcherTest, NoOptionsInCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1;
  std::vector<VectorXf> descriptor2;
  for (int i = 0; i < kNumDescriptors; i++) {
    Eigen::VectorXf rand_vec(kNumDescriptorDimensions);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor1.emplace_back(rand_vec);
    descriptor2.emplace_back(rand_vec);
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
  std::vector<VectorXf> descriptor1;
  std::vector<VectorXf> descriptor2;
  for (int i = 0; i < kNumDescriptors; i++) {
    Eigen::VectorXf rand_vec(kNumDescriptorDimensions);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor1.emplace_back(rand_vec);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor2.emplace_back(rand_vec);
  }

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
  EXPECT_LE(matches[0].correspondences.size(), kNumDescriptors);
}

TEST(FlannFeatureMatcherTest, SymmetricMatchesInCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1;
  std::vector<VectorXf> descriptor2;
  for (int i = 0; i < kNumDescriptors; i++) {
    Eigen::VectorXf rand_vec(kNumDescriptorDimensions);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor1.emplace_back(rand_vec);
    descriptor2.emplace_back(rand_vec);
  }

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
  EXPECT_LE(matches[0].correspondences.size(), kNumDescriptors);
}

TEST(FlannFeatureMatcherTest, NoOptionsOutOfCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1;
  std::vector<VectorXf> descriptor2;
  for (int i = 0; i < kNumDescriptors; i++) {
    Eigen::VectorXf rand_vec(kNumDescriptorDimensions);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor1.emplace_back(rand_vec);
    descriptor2.emplace_back(rand_vec);
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
  std::vector<VectorXf> descriptor1;
  std::vector<VectorXf> descriptor2;
  for (int i = 0; i < kNumDescriptors; i++) {
    Eigen::VectorXf rand_vec(kNumDescriptorDimensions);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor1.emplace_back(rand_vec);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor2.emplace_back(rand_vec);
  }

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
  EXPECT_LE(matches[0].correspondences.size(), kNumDescriptors);
}

TEST(FlannFeatureMatcherTest, SymmetricMatchesOutOfCore) {
  // Set up descriptors.
  std::vector<VectorXf> descriptor1;
  std::vector<VectorXf> descriptor2;
  for (int i = 0; i < kNumDescriptors; i++) {
    Eigen::VectorXf rand_vec(kNumDescriptorDimensions);
    rng->SetRandom(&rand_vec);
    rand_vec.normalize();
    descriptor1.emplace_back(rand_vec);
    descriptor2.emplace_back(rand_vec);
  }

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
  EXPECT_LE(matches[0].correspondences.size(), kNumDescriptors);
}

}  // namespace theia
