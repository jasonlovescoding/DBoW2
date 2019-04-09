/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <memory> 

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// libtorch
#include <torch/script.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

torch::Tensor nms_fast(torch::Tensor in_corners, int height, int width, vector<vector<cv::Mat > > &features, int dist_thresh);
void run(std::shared_ptr<torch::jit::script::Module> module, torch::Tensor input, vector<vector<cv::Mat > > &features);
void loadFeatures(vector<vector<cv::Mat > > &features, int height=120, int width=160);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  testVocCreation(features);

  wait();

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

torch::Tensor nms_fast(torch::Tensor in_corners, int height, int width, vector<vector<cv::Mat > > &features, int dist_thresh)
{
  torch::Tensor grid = torch::zeros({height, width}).to(torch::kInt64);  
  torch::Tensor inds = torch::zeros({height, width}).to(torch::kInt64);  
  // Sort by confidence and round to nearest int.
  torch::Tensor inds1 = torch::argsort(-in_corners.index_select(0, torch::full({1}, 2).to(torch::kLong))).squeeze();
  torch::Tensor corners = in_corners.index_select(1, inds1);
  torch::Tensor rcorners = corners.index_select(0, torch::arange(2).to(torch::kLong)).round().to(torch::kInt64);
  // Check for edge case of 0 or 1 corners.
  if (rcorners.sizes()[1] == 0)
  {
    return torch::zeros({3, 0}).to(torch::kInt64);
  }
  if (rcorners.sizes()[1] == 1)
  {
    return torch::cat({rcorners, in_corners[2]}).reshape({3,1});
  }
  // Initialize the grid.
  for (int i = 0; i < rcorners.sizes()[1]; i++)
  {
    auto col = rcorners[0][i];
    auto row = rcorners[1][i];
    grid[row][col] = 1;
    inds[row][col] = i;
  }
  // Pad the border of the grid, so that we can NMS points near the border.
  int pad = dist_thresh;
  grid = torch::constant_pad_nd(grid, {pad, pad, pad, pad});
  // Iterate through points, highest to lowest conf, suppress neighborhood.
  int count = 0;
  for (int i = 0; i < rcorners.sizes()[1]; i++)
  {
    auto rc = rcorners.index_select(1, torch::full({1}, i).to(torch::kLong));   
    int col = rc[0].item<int>() + pad;
    int row = rc[1].item<int>() + pad;
    if (grid[row][col].item<int>() == 1)
    { // not yet suppressed
      for (int r = row - pad; r < row + pad + 1; r++) {
        for (int c = col - pad; c < col + pad + 1; c++) {
          grid[r][c] = 0;
        }
      }
      grid[row][col] = -1;
      count += 1;
    }
  }
  // Get all surviving -1's and return sorted array of remaining corners.
  torch::Tensor keep = (grid == -1).nonzero();
  auto keepy = keep.index_select(1, torch::zeros(1).to(torch::kLong)).squeeze() - pad;
  auto keepx = keep.index_select(1, torch::ones(1).to(torch::kLong)).squeeze() - pad;
  torch::Tensor inds_keep = torch::zeros(keepx.sizes()[0]).to(torch::kLong);
  for (int i = 0; i < inds_keep.sizes()[0]; i++)
  {
    auto row = keepy[i];
    auto col = keepx[i];
    inds_keep[i] = inds[row][col];
  }
  torch::Tensor out = corners.index_select(1, inds_keep);
  torch::Tensor values = out.index_select(0, torch::full({1}, out.sizes()[0] - 1).to(torch::kLong));
  torch::Tensor inds2 = torch::argsort(-values).to(torch::kLong).squeeze();
  out = out.index_select(1, inds2);
  return out;
}

// ----------------------------------------------------------------------------

void run(std::shared_ptr<torch::jit::script::Module> module, torch::Tensor input, vector<vector<cv::Mat > > &features)
{
  int cell = 8;
  float conf_thresh = 0.015; 
  int nms_dist = 4;
  int H = input.size(2);
  int W = input.size(3); 
  int Hc = H / cell;
  int Wc = W / cell; 
  // int border_remove = 4;

  auto output = module->forward({torch::autograd::make_variable(input, false)}).toTuple();
  torch::Tensor semi = output->elements()[0].toTensor();
  torch::Tensor coarse_desc = output->elements()[1].toTensor();

  // softmax
  semi = semi.squeeze();
  torch::Tensor dense = torch::exp(semi);
  dense = dense / (dense.sum(0) + 0.00001);

  // remove dustbin
  torch::Tensor nodust = dense.slice(0, 0, dense.size(0) - 1);
  nodust = nodust.transpose(0, 1);
  nodust = nodust.transpose(1, 2);

  // get full resolution heatmap
  torch::Tensor heatmap = nodust.reshape({Hc, Wc, cell, cell});
  heatmap = heatmap.transpose(1, 2);
  heatmap = heatmap.reshape({Hc * cell, Wc * cell});

  torch::Tensor ind = (heatmap > conf_thresh).nonzero();
  if (ind.size(0) == 0)
  {
    return;
  }
  torch::Tensor pts = torch::zeros({3, ind.size(0)}); 
  auto xs = ind.index_select(1, torch::zeros(1).to(torch::kLong)).squeeze();
  auto ys = ind.index_select(1, torch::ones(1).to(torch::kLong)).squeeze();
  for (int i = 0; i < ind.size(0); i++)
  {
    pts[0][i] = ys[i];
    pts[1][i] = xs[i];
    pts[2][i] = heatmap[xs[i]][ys[i]];
  }
  pts = nms_fast(pts, H, W, features, nms_dist); 
  torch::Tensor inds = torch::argsort(pts.index_select(0, torch::full({1}, 2).to(torch::kLong)), -1, true).squeeze();
  pts = pts.index_select(1, inds);
  // neglect border removing (a TODO, maybe?)
  torch::Tensor desc;
  int D = coarse_desc.sizes()[1];
  if (pts.sizes()[1] == 0)
  {
    desc = torch::zeros({D, 0});
  }
  else
  {
    auto samp_pts = pts.index_select(0, torch::arange(2).to(torch::kLong)).clone();
    samp_pts[0] = (samp_pts[0] / (float(W)/2.)) - 1.;
    samp_pts[1] = (samp_pts[1] / (float(H)/2.)) - 1.;
    samp_pts = samp_pts.transpose(0, 1).contiguous();
    samp_pts = samp_pts.reshape({1, 1, -1, 2});
    samp_pts = samp_pts.to(torch::kFloat32);
    desc = torch::grid_sampler(coarse_desc, samp_pts, 0, 0);
    desc = desc.reshape({D, -1});
    desc /= torch::frobenius_norm(desc, {0});
  }
  cout << torch::sum(desc) << endl;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features, int height, int width)
{
  features.clear();
  features.reserve(NIMAGES);

  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../demo/superpoint_v1.pt");
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "images/image" << i << ".jpg";

    cv::Mat image = cv::imread(ss.str(), cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    image.convertTo(image, CV_32FC1, 1.0/255.0);

    torch::Tensor input = torch::CPU(torch::kFloat32).tensorFromBlob(image.data, {1, height, width, 1});
    input = input.permute({0,3,1,2});

    // Create a vector of inputs.
    cout << "Extracting superpoint features..." << endl;
    run(module, input, features);
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  OrbVocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


