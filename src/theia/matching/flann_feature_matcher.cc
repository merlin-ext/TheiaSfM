//
// Created by Raul Diaz on 6/1/17.
//

#include <Eigen/Core>
#include <glog/logging.h>
#include <algorithm>
#include <vector>

#include "theia/matching/flann_feature_matcher.h"
#include "theia/matching/feature_matcher_utils.h"
#include "omp.h"

namespace theia {

FlannIndexedImage::FlannIndexedImage(const std::vector<Eigen::VectorXf>& descriptors)
{
  table_ = std::unique_ptr<FlannTable>(new FlannTable(descriptors.size(), descriptors.front().size()));
  for (int i = 0; i < descriptors.size(); i++)
    table_->row(i) = descriptors[i];

  flann_table_ = std::unique_ptr<flann::Matrix<float>>(new flann::Matrix<float>(table_->data(), table_->rows(),
                                                                                table_->cols()));
  index_ = std::unique_ptr<FlannIndex>(new FlannIndex(*flann_table_, flann::KDTreeIndexParams(1)));
  index_->buildIndex();
}

bool FlannFeatureMatcher::MatchImagePair(
        const KeypointsAndDescriptors &features1,
        const KeypointsAndDescriptors &features2,
        std::vector<IndexedFeatureMatch> *matches)
{
  static const int kNumNearestNeighbors = 2;

  const double sq_lowes_ratio =
      this->options_.lowes_ratio * this->options_.lowes_ratio;

  // Compute forward matches.
  // Query the KD-tree to get the top 2 nearest neighbors.
  std::vector<std::vector<float>> nn_distances;
  std::vector<std::vector<int>> nn_indices;
  flann::SearchParams params(128);
  params.cores = 4;

  const FlannIndex* index2 = indexed_images_[features2.image_name].get()->getIndex();
  const flann::Matrix<float>* flann_descriptors1 = indexed_images_[features1.image_name].get()->getTable();
  index2->knnSearch(*flann_descriptors1, nn_indices, nn_distances, kNumNearestNeighbors, params);

  // Output the matches
  for (int i = 0; i < flann_descriptors1->rows; i++) {
    // Add to the matches vector if lowes ratio test is turned off or it is
    // turned on and passes the test.
    if (!this->options_.use_lowes_ratio ||
        nn_distances[i][0] < sq_lowes_ratio * nn_distances[i][1]) {
      matches->emplace_back(IndexedFeatureMatch(i, nn_indices[i][0], nn_distances[i][0]));
    }
  }

  if (matches->size() < this->options_.min_num_feature_matches) {
    return false;
  }

  // Compute the symmetric matches, if applicable.
  if (this->options_.keep_only_symmetric_matches) {
    std::vector<IndexedFeatureMatch> reverse_matches;
    nn_distances.clear();
    nn_indices.clear();

    const FlannIndex* index1 = indexed_images_[features1.image_name].get()->getIndex();
    const flann::Matrix<float>* flann_descriptors2 = indexed_images_[features2.image_name].get()->getTable();
    index1->knnSearch(*flann_descriptors2, nn_indices, nn_distances, kNumNearestNeighbors, params);

    // Output the matches
    for (int i = 0; i < flann_descriptors2->rows; i++) {
      // Add to the matches vector if lowes ratio test is turned off or it is
      // turned on and passes the test.
      if (!this->options_.use_lowes_ratio ||
          nn_distances[i][0] < sq_lowes_ratio * nn_distances[i][1]) {
        reverse_matches.emplace_back(IndexedFeatureMatch(i, nn_indices[i][0], nn_distances[i][0]));
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
    indexed_images_.emplace(image, std::make_shared<FlannIndexedImage>(descriptors));

    VLOG(1) << "Created the kd-tree index for image: " << image;
  }
}

}
