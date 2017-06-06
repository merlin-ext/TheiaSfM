//
// Created by Raul Diaz on 6/1/17.
//

#include <Eigen/Core>
#include <glog/logging.h>
#include <algorithm>
#include <vector>

#include "theia/matching/distance.h"
#include "theia/matching/flann_feature_matcher.h"
#include "theia/matching/feature_matcher_utils.h"
#include "theia/matching/indexed_feature_match.h"

namespace theia {

bool FlannFeatureMatcher::MatchImagePair(
        const KeypointsAndDescriptors &features1,
        const KeypointsAndDescriptors &features2,
        std::vector<IndexedFeatureMatch> *matches)
{
  static const int kNumNearestNeighbors = 2;
  //const flann::Matrix<float>& descriptors1 = image_descriptors_[features1.image_name];
  //const flann::Matrix<float>& descriptors2 = image_descriptors_[features2.image_name];
  const std::vector<Eigen::VectorXf>& descriptors1 = features1.descriptors;
  const std::vector<Eigen::VectorXf>& descriptors2 = features2.descriptors;
  matches->reserve(descriptors1.size());

  const double sq_lowes_ratio =
      this->options_.lowes_ratio * this->options_.lowes_ratio;

  // Compute forward matches.
  matches->reserve(descriptors2.size());

  // Query the KD-tree to get the top 2 nearest neighbors.
  std::vector<std::vector<float>> nn_distances;
  std::vector<std::vector<int>> nn_indices;
  // Gather descriptors
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    eigen_descriptors2(descriptors2.size(), descriptors2.front().size());
  for (int i = 0; i < descriptors2.size(); i++)
    eigen_descriptors2.row(i) = descriptors2[i];
  // Create the descriptor table
  flann::Matrix<float> flann_descriptors2(eigen_descriptors2.data(),
                                          eigen_descriptors2.rows(), eigen_descriptors2.cols());

  const flann::Index<flann::L2<float>>& index1 = indexed_images_.at(features1.image_name);
  index1.knnSearch(flann_descriptors2, nn_indices, nn_distances, kNumNearestNeighbors, flann::SearchParams());

  // Output the matches
  for (int i = 0; i < descriptors2.size(); i++) {
    // Add to the matches vector if lowes ratio test is turned off or it is
    // turned on and passes the test.
    if (!this->options_.use_lowes_ratio ||
        nn_distances[i][0] < sq_lowes_ratio * nn_distances[i][1]) {
      matches->emplace_back(IndexedFeatureMatch(nn_indices[i][0], i, nn_distances[i][0]));
    }
  }

  if (matches->size() < this->options_.min_num_feature_matches) {
    return false;
  }

  // Compute the symmetric matches, if applicable.
  if (this->options_.keep_only_symmetric_matches) {
    std::vector<IndexedFeatureMatch> reverse_matches;
    reverse_matches.reserve(descriptors1.size());
    nn_distances.clear();
    nn_indices.clear();
    // Gather descriptors
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eigen_descriptors1(descriptors1.size(), descriptors1.front().size());
    for (int i = 0; i < descriptors1.size(); i++)
      eigen_descriptors1.row(i) = descriptors1[i];
    // Create the descriptor table
    flann::Matrix<float> flann_descriptors1(eigen_descriptors1.data(),
                                            eigen_descriptors1.rows(), eigen_descriptors1.cols());

    const flann::Index<flann::L2<float>>& index2 = indexed_images_.at(features2.image_name);
    index2.knnSearch(flann_descriptors1, nn_indices, nn_distances, kNumNearestNeighbors, flann::SearchParams());

    // Output the matches
    for (int i = 0; i < descriptors1.size(); i++) {
      // Add to the matches vector if lowes ratio test is turned off or it is
      // turned on and passes the test.
      if (!this->options_.use_lowes_ratio ||
          nn_distances[i][0] < sq_lowes_ratio * nn_distances[i][1]) {
        reverse_matches.emplace_back(IndexedFeatureMatch(nn_indices[i][0], i, nn_distances[i][0]));
      }
    }

    IntersectMatches(reverse_matches, matches);
  }

  return matches->size() >= this->options_.min_num_feature_matches;
}

void FlannFeatureMatcher::AddImage(
    const std::string& image,
    const std::vector<Keypoint>& keypoints,
    const std::vector<Eigen::VectorXf>& descriptors)
{
  // This will save the descriptors and keypoints to disk and set up our LRU
  // cache.
  FeatureMatcher::AddImage(image, keypoints, descriptors);

  // Create the kd-tree.
  if (!ContainsKey(indexed_images_, image)) {
    // Gather the descriptors.
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      eigen_descriptors(descriptors.size(),
                            descriptors.front().size());
    for (int i = 0; i < descriptors.size(); i++)
      eigen_descriptors.row(i) = descriptors[i];

    // Create the descriptor table
    flann::Matrix<float> flann_descriptors(
        eigen_descriptors.data(), eigen_descriptors.rows(),
        eigen_descriptors.cols());

    // Create the searchable KD-tree with FLANN.
    flann::Index<flann::L2<float> > flann_kd_tree(
        flann_descriptors, flann::KDTreeSingleIndexParams());
    flann_kd_tree.buildIndex();
    indexed_images_.emplace(image, flann_kd_tree);

    VLOG(1) << "Created the kd-tree index for image: " << image;
  }
}

}