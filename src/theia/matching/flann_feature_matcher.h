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

// Performs features matching between two sets of features using a cascade
// hashing approach. This hashing does not require any training and is extremely
// efficient but can only be used with float features like SIFT.
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

 private:
  bool MatchImagePair(const KeypointsAndDescriptors& features1,
                      const KeypointsAndDescriptors& features2,
                      std::vector<IndexedFeatureMatch>* matches) override;

  std::unordered_map<std::string, flann::Index<flann::L2<float>>>  indexed_images_;

  DISALLOW_COPY_AND_ASSIGN(FlannFeatureMatcher);
};

}  // namespace theia

#endif //THEIA_FLANN_FEATURE_MATCHER_H
