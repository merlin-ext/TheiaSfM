//
// Created by Raul Diaz on 6/1/17.
//

#ifndef THEIA_FLANN_FEATURE_MATCHER_H
#define THEIA_FLANN_FEATURE_MATCHER_H

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "theia/matching/feature_matcher.h"
#include "theia/util/util.h"
#include "flann/flann.hpp"

namespace theia {
    class Keypoint;
    struct CameraIntrinsicsPrior;
    struct IndexedFeatureMatch;
    struct KeypointsAndDescriptors;

    typedef flann::Index<flann::L2<float>> FlannIndex;
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> FlannTable;

// Simple container class to hold a FLANN index and a table structure to query features
class FlannIndexedImage
{
 public:
  FlannIndexedImage(const std::vector<Eigen::VectorXf>& descriptors);

  const FlannIndex* getIndex() { return index_.get(); }

  const flann::Matrix<float>* getTable() { return flann_table_.get(); }

 private:
  std::unique_ptr<FlannTable> table_; // Eigen table, will be destroyed safely
  std::unique_ptr<FlannIndex> index_; // Flann index to perform searches
  std::unique_ptr<flann::Matrix<float>> flann_table_; // Flann data wrapper. No need to destroy
};

// Performs features matching between two sets of features using a FLANN
// indexing approach.
class FlannFeatureMatcher : public FeatureMatcher {
 public:
  explicit FlannFeatureMatcher(const FeatureMatcherOptions& options)
          : FeatureMatcher(options) {}
  ~FlannFeatureMatcher() {}

  // These methods are the same as the base class except that the HashedImage is
  // created as the descriptors are added.
  void AddImage(const std::string& image,
                const std::vector<Keypoint>& keypoints,
                const std::vector<Eigen::VectorXf>& descriptors) override;

  void AddImage(const std::string& image_name,
                const std::vector<Keypoint>& keypoints,
                const std::vector<Eigen::VectorXf>& descriptors,
                const CameraIntrinsicsPrior& intrinsics) override;

 private:
  bool MatchImagePair(const KeypointsAndDescriptors& features1,
                      const KeypointsAndDescriptors& features2,
                      std::vector<IndexedFeatureMatch>* matches) override;

  std::unordered_map<std::string, std::shared_ptr<FlannIndexedImage> > indexed_images_;
  std::mutex indexed_images_lock_; // locks the addition of data to indexed_images_, image_names_, and intrinsics_

  DISALLOW_COPY_AND_ASSIGN(FlannFeatureMatcher);
};

}  // namespace theia

#endif //THEIA_FLANN_FEATURE_MATCHER_H
