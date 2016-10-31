/**
 * Sammon Mapping with the GNU Scientific Library
 * Chris Rayner (2013; cleaned up 2016)
 */

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

// Preprocessor options:
// whether to use Newton's method to compute the search direction
// #define USE_NEWTON
// Taylor approx for backtracking Armijo/Goldstein linesearch
// #define USE_SMART_LINE

const int MAX_ITERATIONS = 4000;
const int MAX_BACKTRACKS = 50;
const double TOLERANCE = 1e-5;

const int VERBOSE = 1; // show progress bar, error reporting, etc.

const double BACKTRACK_AMOUNT = .5; // step-reduction amount

gsl_matrix * tmp_matrix; // extra variable to dump temporary values

size_t target_dim; // dimensionality of the target space

/// Utility functions
///==================

void progress_bar(int ii, int imax) {
  static int previous_progress = -1;
  const int bar_width = 20;
  int how_much = (int) round(ii * bar_width/imax);
  if (previous_progress != how_much) {
    fprintf(stderr, "[");
    int kk;
    for (kk = 0; kk < how_much; ++ kk)
      fprintf(stderr, "*");
    for (; kk < bar_width; ++ kk)
      fprintf(stderr, "-");
    fprintf(stderr, "]\r");
  }
  previous_progress = how_much;
}

int is_symmetric(gsl_matrix * D) {
  size_t ii;
  for (ii = 0; ii < D->size1; ++ ii) {
    size_t jj;
    for (jj = ii + 1; jj < D->size1; ++ jj) {
      if (fabs(gsl_matrix_get(D, ii, jj) - gsl_matrix_get(D, jj, ii)) > 1e-5) {
        fprintf(stderr, "ERROR - asymmetry at %zu,%zu: %f != %f", ii, jj,
                gsl_matrix_get(D, ii, jj), gsl_matrix_get(D, jj, ii));
        return 0;
      }
    }
  }
  return 1;
}

gsl_matrix* random_matrix(size_t rows, size_t cols) {
  gsl_matrix * A = gsl_matrix_alloc(rows, cols);
  size_t ii;
  for (ii = 0; ii < A->size1; ++ ii) {
    size_t jj;
    for (jj = 0; jj < A->size2; ++ jj) {
      double val = 1. * rand() / (double) INT_MAX;
      gsl_matrix_set(A, ii, jj, val);
    }
  }
  return A;
}

/// Sammon Mapping
///===============

double euclidean_distance(int ii, int jj, const gsl_matrix * X) {
  // Euclidean distance between rows ii and jj of X.
  gsl_vector_const_view rowii = gsl_matrix_const_row(X, ii);
  gsl_vector_const_view rowjj = gsl_matrix_const_row(X, jj);
  double sum = 0;
  size_t kk;
  for (kk = 0; kk < X->size2; ++ kk) {
    double val = rowii.vector.data[kk] - rowjj.vector.data[kk];
    sum = sum + val * val;
  }
  return sqrt(sum);
}

void compute_distance_matrix(gsl_matrix * D, const gsl_matrix * X) {
  // compute matrix D of pairwise distances between all points in
  // X (with 1s on diagonal). NOTE: accounts for > 50% of runtime
  for (size_t ii = 0; ii < X->size1; ++ ii) {
    gsl_matrix_set(D, ii, ii, 1);
    for (size_t jj = ii + 1; jj < X->size1; ++ jj) {
      double distance = euclidean_distance(ii, jj, X);
      gsl_matrix_set(D, ii, jj, distance);
      gsl_matrix_set(D, jj, ii, distance);
    }
  }
}

gsl_matrix* pairwise_weight_matrix(const gsl_matrix * D) {
  // Sammon's weight matrix
  // the farther apart two points are in input space, the less important it is
  // we get them right
  size_t num_points = D->size1;
  gsl_matrix * W = gsl_matrix_alloc(num_points, num_points);
  gsl_matrix_set_all(W, 1.);
  gsl_matrix_div_elements(W, D);
  for (size_t ii = 0; ii < num_points; ++ ii)
    gsl_matrix_set(W, ii, ii, 1);
  return W;
}

double compute_loss(gsl_matrix * delta, const gsl_matrix * W) {
  // Sammon's loss function
  // NOTE: modifies argument 'delta' for use in computation
  double sum = 0;
  gsl_matrix_mul_elements(delta, delta);
  gsl_matrix_mul_elements(delta, W);
  size_t ii;
  for (ii = 0; ii < delta->size1; ++ ii) {
    size_t jj;
    for (jj = ii + 1; jj < delta->size1; ++ jj)
      sum += gsl_matrix_get(delta, ii, jj);
  }
  return sum;
}

void compute_gradient(gsl_matrix * gradient,
                      const gsl_matrix * delta,
                      const gsl_matrix * Y,
                      const gsl_matrix * delta_ones) {
  // compute the first derivative of Sammon's loss function
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, delta, Y, 0.0, gradient);
  gsl_matrix_memcpy(tmp_matrix, Y);
  gsl_matrix_mul_elements(tmp_matrix, delta_ones);
  gsl_matrix_sub(gradient, tmp_matrix);
}

#ifdef USE_NEWTON
void compute_hessian_diagonal(gsl_matrix * hessian_diag,
                              gsl_matrix * Dinv3,
                              gsl_matrix * Y2,
                              gsl_matrix * delta_ones,
                              gsl_matrix * Y,
                              const gsl_matrix * ones) {
  // compute the diagonal of the second derivative of Sammon's loss function
  // NOTE: modifies argument 'delta_ones' for use in computation
  gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, Dinv3, Y2, 0.0, hessian_diag);
  gsl_matrix_sub(hessian_diag, delta_ones);
  gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, Dinv3, ones, 0.0, delta_ones);
  gsl_matrix_mul_elements(delta_ones, Y2);
  gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, Dinv3, Y, 0.0, tmp_matrix);
  gsl_matrix_mul_elements(tmp_matrix, Y);
  gsl_matrix_scale(tmp_matrix, -2.);
  gsl_matrix_add(hessian_diag, delta_ones);
  gsl_matrix_add(hessian_diag, tmp_matrix);
}

void compute_newton_search_direction(gsl_matrix * search_direction,
                                     const gsl_matrix * gradient,
                                     const gsl_matrix * hessian_diag) {
  int ii;
  for (ii = 0; ii < search_direction->size1; ++ ii) {
    int jj;
    for (jj = 0; jj < search_direction->size2; ++ jj)
      gsl_matrix_set(search_direction, ii, jj,
                     -gsl_matrix_get(gradient, ii, jj)
                     / fabs(gsl_matrix_get(hessian_diag, ii, jj)));
  }
}
#endif

int test_stopping_conditions(double loss, double loss_previous,
                             size_t iteration, size_t backtracks) {
  if (loss < TOLERANCE) {
    if (VERBOSE)
      fprintf(stderr, "Perfect within tolerance (loss %f, iteration %zu)\n",
              loss, iteration);
    return 1;
  }
  if (fabs((loss_previous - loss) / loss_previous) < TOLERANCE) {
    if (VERBOSE)
      fprintf(stderr, "Converged within tolerance (loss %f, iteration %zu)\n",
              loss, iteration);
    return 1;
  }
  if (backtracks == MAX_BACKTRACKS) {
    if (VERBOSE)
      fprintf(stderr, "Stuck (loss %f, iteration %zu)\n",
              loss, iteration);
    return 1;
  }
  if (iteration == MAX_ITERATIONS - 1) {
    if (VERBOSE)
      fprintf(stderr, "Exhausted iterations (loss %f, iteration %zu)\n",
              loss, iteration);
    return 1;
  }
  return 0;
}

gsl_matrix* sammon_mapping(const gsl_matrix * D_goal, size_t target_dim) {
  size_t num_points = D_goal->size1;
  // initialize mapping randomly
  gsl_matrix * Y = random_matrix(num_points, target_dim);
  gsl_matrix * W = pairwise_weight_matrix(D_goal);

  // iteration variables
  gsl_matrix * D           = gsl_matrix_alloc(num_points, num_points);
  gsl_matrix * delta       = gsl_matrix_alloc(num_points, num_points);
  gsl_matrix * delta_ones  = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix * gradient    = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix * increment   = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix * Y_previous  = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix * ones        = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix_set_all(ones, 1);
  tmp_matrix               = gsl_matrix_alloc(num_points, target_dim);

#ifdef USE_NEWTON
  // iteration variables for using Newton's method
  gsl_matrix * hessian_diag = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix * YY           = gsl_matrix_alloc(num_points, target_dim);
  gsl_matrix * Dinv3        = gsl_matrix_alloc(num_points, num_points);
#endif

  double loss, loss_previous;

  // compute loss between current D and D_goal
  compute_distance_matrix(D, Y);
  gsl_matrix_memcpy(delta, D_goal);
  gsl_matrix_sub(delta, D);
  loss = loss_previous = compute_loss(delta, W);

  for (size_t iteration = 0; iteration < MAX_ITERATIONS; ++ iteration) {
    // Compute a search direction in which to increment Y:
    gsl_matrix_memcpy(delta, D_goal);
    gsl_matrix_div_elements(delta, D);
    gsl_matrix_add_constant(delta, -1);
    gsl_matrix_mul_elements(delta, W);
    gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, delta, ones, 0.0, delta_ones);

#ifndef USE_NEWTON
    // use steepest descent as the search direction
    compute_gradient(increment, delta, Y, delta_ones);
    gsl_matrix_scale(increment, -1);
#else
    // use Newton's method to inform the search direction
    compute_gradient(gradient, delta, Y, delta_ones);
    gsl_matrix_memcpy(YY, Y);
    gsl_matrix_mul_elements(YY, Y);
    gsl_matrix_set_all(Dinv3, 1.0);
    gsl_matrix_div_elements(Dinv3, D);
    gsl_matrix_div_elements(Dinv3, D);
    gsl_matrix_div_elements(Dinv3, D);
    compute_hessian_diagonal(hessian_diag, Dinv3, YY, delta_ones, Y, ones);
    compute_newton_search_direction(increment, gradient, hessian_diag);
#endif

    // Determine how far to follow the increment.
#ifdef USE_SMART_LINE
    gsl_matrix_memcpy(gradient, increment);
    gsl_matrix_scale(gradient, -1);
    gsl_vector_const_view search_vector =
      gsl_vector_const_view_array(increment->data,
                                  increment->size1 * increment->size2);
    gsl_vector_const_view gradient_vector =
      gsl_vector_const_view_array(gradient->data,
                                  gradient->size1 * gradient->size2);
#endif

    gsl_matrix_memcpy(Y_previous, Y);
    int backtracks = 0;
    for (; backtracks < MAX_BACKTRACKS; ++ backtracks) {
      // Y = Y_previous + increment
      gsl_matrix_memcpy(Y, Y_previous);
      gsl_matrix_add(Y, increment);

      // compute loss
      compute_distance_matrix(D, Y);
      gsl_matrix_memcpy(delta, D_goal);
      gsl_matrix_sub(delta, D);
      loss = compute_loss(delta, W);

#ifdef USE_SMART_LINE
      double alpha = 1.0, beta = 1e-3;
      double gTp;
      gsl_blas_ddot(&search_vector.vector, &gradient_vector.vector, &gTp);
      if (loss < loss_previous + alpha * beta * gTp)
        break;
      alpha = alpha * BACKTRACK_AMOUNT;
#else
      if (loss < loss_previous)
        break;
#endif

      // shrink the increment down -- it was too much
      gsl_matrix_scale(increment, BACKTRACK_AMOUNT);
    }

    if (VERBOSE)
      progress_bar(iteration, MAX_ITERATIONS);

    if (test_stopping_conditions(loss, loss_previous, iteration, backtracks))
      break;
    loss_previous = loss;
  }

  gsl_matrix_free(W);
  gsl_matrix_free(D);
  gsl_matrix_free(delta);
  gsl_matrix_free(delta_ones);
  gsl_matrix_free(gradient);
  gsl_matrix_free(increment);
  gsl_matrix_free(Y_previous);
  gsl_matrix_free(ones);
  gsl_matrix_free(tmp_matrix);

#ifdef USE_NEWTON
  gsl_matrix_free(hessian_diag);
  gsl_matrix_free(YY);
  gsl_matrix_free(Dinv3);
#endif

  return Y;
}

/// Input/output
///=============

void print_matrix(FILE * stream, gsl_matrix * X) {
  size_t ii;
  for (ii = 0; ii < X->size1; ++ ii) {
    if (ii == 10 && X->size1 > 30 && (stream == stdout || stream == stderr)) {
      fprintf(stream, " |:\n");
      ii = X->size1 - 10;
    }
    size_t jj;
    for (jj = 0; jj < X->size2; ++ jj) {
      if (jj == 3 && X->size2 > 10 && (stream == stdout || stream == stderr)) {
        fprintf(stream, " ... ");
        jj = X->size2 - 3;
      }
      fprintf(stream, " %8f ", gsl_matrix_get(X, ii, jj));
    }
    fprintf(stream, "\n");
  }
  fprintf(stream, "\n");
}

void save_matrix(gsl_matrix * X, char * filename) {
  FILE * output = fopen(filename, "w");
  print_matrix(output, X);
  fclose(output);
  //fprintf(stdout, ":scatter %s:\n", filename);
}

void load_error(FILE * stream) {
  fclose(stream);
  fprintf(stderr, "ERROR - a misformatted input file?\n");
  exit(1);
}

gsl_matrix* load_input_file(char * filename) {
  size_t num_points;
  gsl_matrix * D_goal;
  FILE * stream = fopen(filename, "r");
  if (!stream) {
    fprintf(stderr, "Could not open '%s'\n", filename);
    exit(1);
  }
  char token[100];
  fscanf(stream, "%s", token);
  if (!strcmp("num_points", token)) {
    fscanf(stream, "%s", token);
    num_points = atoi(token);
    assert(num_points > 0);
  }
  else
    load_error(stream);
  fscanf(stream, "%s", token);
  if (!strcmp("target_dim", token)) {
    fscanf(stream, "%s", token);
    target_dim = atoi(token);
    assert(target_dim > 0);
  }
  else
    load_error(stream);
  D_goal = gsl_matrix_alloc(num_points, num_points);

  fscanf(stream, "%s", token);
  if (!strcmp("type_of_data", token)) {
    fscanf(stream, "%s", token);
    if (!strcmp("distances", token)) {
      gsl_matrix_fscanf(stream, D_goal);
      assert(is_symmetric(D_goal));
    }
    else if (!strcmp("points", token)) {
      fscanf(stream, "%s", token);
      size_t source_dim = atoi(token);
      assert(source_dim > target_dim);
      gsl_matrix * X = gsl_matrix_alloc(num_points, source_dim);
      gsl_matrix_fscanf(stream, X);
      compute_distance_matrix(D_goal, X);
      gsl_matrix_free(X);
    }
    else
      load_error(stream);
  }
  else
    load_error(stream);
  fclose(stream);
  return D_goal;
}

int main(int argc, char ** argv) {
  if (argc == 1) {
    fprintf(stderr, "%s <input_file>\n", argv[0]);
    exit(1);
  }
  srand(time(0));
  gsl_matrix * D_goal = load_input_file(argv[1]);
  gsl_matrix * Y = sammon_mapping(D_goal, target_dim);
  save_matrix(Y, "embedding~");
  gsl_matrix_free(D_goal);
  gsl_matrix_free(Y);
  return 0;
}
