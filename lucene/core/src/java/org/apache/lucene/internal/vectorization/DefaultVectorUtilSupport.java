/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.internal.vectorization;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.lucene.util.Constants;
import org.apache.lucene.util.SuppressForbidden;

final class DefaultVectorUtilSupport implements VectorUtilSupport {

  private final FloatApproximateDotProductFunction approxDotProdFunction;

  DefaultVectorUtilSupport() {
    String adpa = System.getProperty("approximate-dotproduct");
    if (adpa == null || adpa.isEmpty()) {
      this.approxDotProdFunction = new CompositeApprox();
    } else {
      FloatApproximateDotProductFunction approximateDotProduct = switch (adpa) {
          case "meansum" -> new MeanSumApproximateDotProductFunction();
          case "xornet" -> new XorNetApproximateDotProductFunction();
          case "axornet" -> new ApproximateXorNetApproximateDotProductFunction();
          case "randproj" -> new RandomProjectionsApproximateDotProductFunction();
          case "norm1bound" -> new Norm1BoundApproximateDotProductFunction();
          case "norm2bound" -> new Norm2BoundApproximateDotProductFunction();
          case "boundmean" -> new BoundMeanApproximateDotProductFunction();
          default -> new MeanSumApproximateDotProductFunction();
      };
        this.approxDotProdFunction = approximateDotProduct;
    }
  }

  // the way FMA should work! if available use it, otherwise fall back to mul/add
  @SuppressForbidden(reason = "Uses FMA only where fast and carefully contained")
  private static float fma(float a, float b, float c) {
    if (Constants.HAS_FAST_SCALAR_FMA) {
      return Math.fma(a, b, c);
    } else {
      return a * b + c;
    }
  }

  @Override
  public float dotProduct(float[] a, float[] b) {
    float res = 0f;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;
      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        acc1 = fma(a[i], b[i], acc1);
        acc2 = fma(a[i + 1], b[i + 1], acc2);
        acc3 = fma(a[i + 2], b[i + 2], acc3);
        acc4 = fma(a[i + 3], b[i + 3], acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    for (; i < a.length; i++) {
      res = fma(a[i], b[i], res);
    }
    return res;
  }

  @Override
  public float approximateDotProduct(float[] a, float[] b) {
    return this.approxDotProdFunction.approximateDotProduct(a, b);
  }

  @Override
  public float cosine(float[] a, float[] b) {
    float sum = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float sum1 = 0;
      float sum2 = 0;
      float norm1_1 = 0;
      float norm1_2 = 0;
      float norm2_1 = 0;
      float norm2_2 = 0;

      int upperBound = a.length & ~(2 - 1);
      for (; i < upperBound; i += 2) {
        // one
        sum1 = fma(a[i], b[i], sum1);
        norm1_1 = fma(a[i], a[i], norm1_1);
        norm2_1 = fma(b[i], b[i], norm2_1);

        // two
        sum2 = fma(a[i + 1], b[i + 1], sum2);
        norm1_2 = fma(a[i + 1], a[i + 1], norm1_2);
        norm2_2 = fma(b[i + 1], b[i + 1], norm2_2);
      }
      sum += sum1 + sum2;
      norm1 += norm1_1 + norm1_2;
      norm2 += norm2_1 + norm2_2;
    }

    for (; i < a.length; i++) {
      sum = fma(a[i], b[i], sum);
      norm1 = fma(a[i], a[i], norm1);
      norm2 = fma(b[i], b[i], norm2);
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public float approximateCosine(float[] v1, float[] v2) {
    throw new UnsupportedOperationException();
  }

  @Override
  public float squareDistance(float[] a, float[] b) {
    float res = 0;
    int i = 0;

    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;

      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        // one
        float diff1 = a[i] - b[i];
        acc1 = fma(diff1, diff1, acc1);

        // two
        float diff2 = a[i + 1] - b[i + 1];
        acc2 = fma(diff2, diff2, acc2);

        // three
        float diff3 = a[i + 2] - b[i + 2];
        acc3 = fma(diff3, diff3, acc3);

        // four
        float diff4 = a[i + 3] - b[i + 3];
        acc4 = fma(diff4, diff4, acc4);
      }
      res += acc1 + acc2 + acc3 + acc4;
    }

    for (; i < a.length; i++) {
      float diff = a[i] - b[i];
      res = fma(diff, diff, res);
    }
    return res;
  }

  /**
   * Approximate euclidean distance from "Speeding up k-means by approximating Euclidean distances via block vectors"
   * by Bottesch et al., 2016.
   * {@see <a href="https://proceedings.mlr.press/v48/bottesch16.pdf"/>}
   * @param a dense vector
   * @param b dense vector
   * @return approximate euclidean distance
   */
  @Override
  public float approximateSquareDistance(float[] a, float[] b) {
    float aNorm = 0;
    float bNorm = 0;
    int blockSize = 8;
    int reducedSize = a.length / blockSize;
    float[] aBlocked = new float[reducedSize];
    float[] bBlocked = new float[reducedSize];

    // TODO : build blocked arrays

    int i = 0;
    // if the array is big, unroll it
    if (a.length > 32) {
      float acc1 = 0;
      float acc2 = 0;
      float acc3 = 0;
      float acc4 = 0;
      float acc5 = 0;
      float acc6 = 0;
      float acc7 = 0;
      float acc8 = 0;

      int upperBound = a.length & ~(4 - 1);
      for (; i < upperBound; i += 4) {
        // one
        acc1 = fma(a[i], a[i], acc1);
        acc5 = fma(b[i], b[i], acc5);

        // two
        acc2 = fma(a[i + 1], a[i + 1], acc2);
        acc6 = fma(b[i + 1], b[i + 1], acc6);

        // three
        acc3 = fma(a[i + 2], a[i + 2], acc3);
        acc7 = fma(b[i + 2], b[i + 2], acc7);

        // four
        acc4 = fma(a[i + 3], a[i + 3], acc4);
        acc8 = fma(b[i + 3], b[i + 3], acc8);
      }
      aNorm += acc1 + acc2 + acc3 + acc4;
      bNorm += acc5 + acc6 + acc7 + acc8;
    }

    for (; i < a.length; i++) {
      aNorm = fma(a[i], a[i], aNorm);
      bNorm = fma(b[i], b[i], bNorm);
    }
    return approxSquareRoot(
        fma(
            aNorm,
            aNorm,
            approxSquareRoot(fma(bNorm, bNorm, -2 * (approximateDotProduct(aBlocked, bBlocked))))));
  }

  @Override
  public int dotProduct(byte[] a, byte[] b) {
    int total = 0;
    for (int i = 0; i < a.length; i++) {
      total += a[i] * b[i];
    }
    return total;
  }

  @Override
  public int approximateDotProduct(byte[] a, byte[] b) {
    throw new UnsupportedOperationException();
  }

  @Override
  public float cosine(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int sum = 0;
    int norm1 = 0;
    int norm2 = 0;

    for (int i = 0; i < a.length; i++) {
      byte elem1 = a[i];
      byte elem2 = b[i];
      sum += elem1 * elem2;
      norm1 += elem1 * elem1;
      norm2 += elem2 * elem2;
    }
    return (float) (sum / Math.sqrt((double) norm1 * (double) norm2));
  }

  @Override
  public float approximateCosine(byte[] a, byte[] b) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int squareDistance(byte[] a, byte[] b) {
    // Note: this will not overflow if dim < 2^18, since max(byte * byte) = 2^14.
    int squareSum = 0;
    for (int i = 0; i < a.length; i++) {
      int diff = a[i] - b[i];
      squareSum += diff * diff;
    }
    return squareSum;
  }

  @Override
  public int approximateSquareDistance(byte[] a, byte[] b) {
    throw new UnsupportedOperationException();
  }

  /**
   * A function to calculate approximate (faster) dot product.
   */
  interface FloatApproximateDotProductFunction {
    float approximateDotProduct(float[] a, float[] b);
  }

  /**
   * Calculate dot product by calculating multiple dot products over smaller-masked-filtered subvectors.
   */
  private class RandomProjectionsApproximateDotProductFunction
      implements FloatApproximateDotProductFunction {

    private int dims;
    private int[][] masks;
    private final SecureRandom random = new SecureRandom();

    RandomProjectionsApproximateDotProductFunction() {
      this.masks = new int[4][8];
      this.dims = 100;
      initMasks();
    }

    private void initMasks() {
      this.masks[0] = random.ints(8, 0, dims).toArray();
      this.masks[1] = random.ints(8, 0, dims).toArray();
      this.masks[2] = random.ints(8, 0, dims).toArray();
      this.masks[3] = random.ints(8, 0, dims).toArray();
    }

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      float res;
      // if the array is big, unroll it
      if (a.length > 32) {
        if (this.dims != a.length) {
          this.dims = a.length;
          initMasks();
        }
        res = 1;
        for (int[] mask : masks) {
          float projRes = 0;
          for (int j : mask) {
            projRes = fma(a[j], b[j], projRes);
          }
          res *= projRes;
        }
      } else {
        res = 0;
        for (int i = 0; i < a.length; i++) {
          res = fma(a[i], b[i], res);
        }
      }

      return res;
    }
  }

  private class VectorQuantizationApproximateDotProductFunction implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      float res = 1f;
      int i = 0;
      int jump = 64;

      // if the array is big, unroll it
      if (a.length > 32) {
        int upperBound = a.length & ~(jump - 1);
        for (; i < upperBound; i += jump) {
          res *= dotProduct(Arrays.copyOfRange(a, i, jump+i), Arrays.copyOfRange(b, i, jump+i));
        }
      }

      if (i < a.length) {
        res *= dotProduct(Arrays.copyOfRange(a, i, a.length-1), Arrays.copyOfRange(b, i, a.length-1));
      }
      return res;
    }
  }

  /**
   * Approximate dot product by vectors' sum, when vectors are close enough (should mostly be the case for HNSW candidates).
   * {@see <a href="https://math.stackexchange.com/questions/2863282/approximate-scalar-dot-product-with-a-vectors-sum"/>}
   */
  private class MeanSumApproximateDotProductFunction implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      int i = 0;
      float sum1 = 0;
      float sum2 = 0;
      // if the array is big, unroll it
      if (a.length > 32) {
        float acc1 = 0;
        float acc2 = 0;
        float acc3 = 0;
        float acc4 = 0;

        float acc5 = 0;
        float acc6 = 0;
        float acc7 = 0;
        float acc8 = 0;

        int upperBound = a.length & ~(4 - 1);
        for (; i < upperBound; i += 4) {
          // one
          acc1 += a[i];
          acc5 += b[i];

          // two
          acc2 += a[i + 1];
          acc6 += b[i + 1];

          // three
          acc3 += a[i + 2];
          acc7 += b[i + 2];

          // four
          acc4 += a[i + 3];
          acc8 += b[i + 3];
        }
        sum1 += acc1 + acc2 + acc3 + acc4;
        sum2 += acc5 + acc6 + acc7 + acc8;
      }

      for (; i < a.length; i++) {
        sum1 += a[i];
        sum2 += b[i];
      }
      return sum2 * (sum1 / a.length);
    }
  }

  /**
   * Two-norm upper bound based approximation of dot product.
   * UB = (||v1||^2 + ||v2||^2) / 2
   * {@see <a href="https://math.stackexchange.com/questions/4670314/upper-bound-of-the-dot-product-of-two-real-valued-vectors"/>}
   */
  private class Norm2BoundApproximateDotProductFunction
      implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      int i = 0;
      float normA = 0;
      float normB = 0;

      // if the array is big, unroll it
      if (a.length > 32) {
        float accA1 = 0;
        float accB1 = 0;

        float accA2 = 0;
        float accB2 = 0;

        float accA3 = 0;
        float accB3 = 0;

        float accA4 = 0;
        float accB4 = 0;

        int upperBound = a.length & ~(4 - 1);
        for (; i < upperBound; i += 4) {
          // one
          accA1 = fma(a[i], a[i], accA1);
          accB1 = fma(b[i], b[i], accB1);

          // two
          accA2 = fma(a[i + 1], a[i + 1], accA2);
          accB2 = fma(b[i + 1], b[i + 1], accB2);

          // three
          accA3 = fma(a[i + 2], a[i + 2], accA3);
          accB3 = fma(b[i + 2], b[i + 2], accB3);

          // four
          accA4 = fma(a[i + 3], a[i + 3], accA4);
          accB4 = fma(b[i + 3], b[i + 3], accB4);
        }
        normA = accA1 + accA2 + accA3 + accA4;
        normB = accB1 + accB2 + accB3 + accB4;
      }

      for (; i < a.length; i++) {
        normA = fma(a[i], a[i], normA);
        normB = fma(b[i], b[i], normB);
      }

      return (float) ((Math.sqrt(normA) * Math.sqrt(normB)));
    }
  }

  /**
   * One-norm upper bound based approximation of dot product.
   * UB = (|v1| + |v2|) / 2
   */
  private class Norm1BoundApproximateDotProductFunction
          implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      int i = 0;
      float normA = 0;
      float normB = 0;

      // if the array is big, unroll it
      if (a.length > 32) {
        float accA1 = 0;
        float accB1 = 0;

        float accA2 = 0;
        float accB2 = 0;

        float accA3 = 0;
        float accB3 = 0;

        float accA4 = 0;
        float accB4 = 0;

        int upperBound = a.length & ~(4 - 1);
        for (; i < upperBound; i += 4) {
          // one
          accA1 += Math.abs(a[i]);
          accB1 += Math.abs(b[i]);

          // two
          accA2 += Math.abs(a[i+1]);
          accB2 += Math.abs(a[i+1]);

          // three
          accA3 += Math.abs(a[i+2]);
          accB3 += Math.abs(b[i+2]);

          // four
          accA4 = Math.abs(a[i+3]);
          accB4 = Math.abs(b[i+3]);;
        }
        normA = accA1 + accA2 + accA3 + accA4;
        normB = accB1 + accB2 + accB3 + accB4;
      }

      for (; i < a.length; i++) {
        normA += Math.abs(a[i]);
        normB += Math.abs(b[i]);
      }

      return (normA + normB) / 2;
    }
  }

  /**
   * XORNet dot product approximation, {@see "Binary Graph Neural Networks", Bahri et al., 2021}.
   */
  private class XorNetApproximateDotProductFunction implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      float res = 0;
      int i = 0;
      float normA = 0;
      float normB = 0;

      // if the array is big, unroll it
      if (a.length > 32) {
        float accA1 = 0;
        float accB1 = 0;
        float accR1 = 0;

        float accA2 = 0;
        float accB2 = 0;
        float accR2 = 0;

        float accA3 = 0;
        float accB3 = 0;
        float accR3 = 0;

        float accA4 = 0;
        float accB4 = 0;
        float accR4 = 0;

        int upperBound = a.length & ~(4 - 1);
        for (; i < upperBound; i += 4) {
          // one
          accA1 += Math.abs(a[i]);
          accB1 += Math.abs(b[i]);
          accR1 = fma(Math.signum(a[i]), Math.signum(b[i]), accR1);

          // two
          accA2 += Math.abs(a[i + 1]);
          accB2 += Math.abs(b[i + 1]);
          accR2 = fma(Math.signum(a[i + 1]), Math.signum(b[i + 1]), accR2);

          // three
          accA3 += Math.abs(a[i + 2]);
          accB3 += Math.abs(b[i + 2]);
          accR3 = fma(Math.signum(a[i + 2]), Math.signum(b[i + 2]), accR3);

          // four
          accA4 += Math.abs(a[i + 3]);
          accB4 += Math.abs(b[i + 3]);
          accR4 = fma(Math.signum(a[i + 3]), Math.signum(b[i + 3]), accR4);
        }
        normA += accA1 + accA2 + accA3 + accA4;
        normB += accB1 + accB2 + accB3 + accB4;
        res += accR1 + accR2 + accR3 + accR4;
      }

      for (; i < a.length; i++) {
        normA += Math.abs(a[i]);
        normB += Math.abs(b[i]);
        res = fma(Math.signum(a[i]), Math.signum(b[i]), res);
      }

      return res * (normA / a.length) * (normB / a.length);
    }
  }

  /**
   * Approximate XORNet dot product approximation, {@see "Binary Graph Neural Networks", Bahri et al., 2021},
   * where {@code sign(v1)*sign(v2)} becomes {@code sign(v1) == sign(v2)}.
   */
  private class ApproximateXorNetApproximateDotProductFunction
      implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      float res = 0;
      int i = 0;
      float normA = 0;
      float normB = 0;
      float a1;
      float a2;

      // if the array is big, unroll it
      if (a.length > 32) {
        float accA1 = 0;
        float accB1 = 0;
        float accR1 = 0;

        float accA2 = 0;
        float accB2 = 0;
        float accR2 = 0;

        float accA3 = 0;
        float accB3 = 0;
        float accR3 = 0;

        float accA4 = 0;
        float accB4 = 0;
        float accR4 = 0;

        int upperBound = a.length & ~(4 - 1);
        for (; i < upperBound; i += 4) {
          a1 = a[i];
          a2 = b[i];
          // one
          accA1 += Math.abs(a1);
          accB1 += Math.abs(a2);
          accR1 += Math.signum(a1) == Math.signum(a2) ? 1 : 0;

          // two
          a1 = a[i + 1];
          a2 = b[i + 1];
          accA2 += Math.abs(a1);
          accB2 += Math.abs(a2);
          accR2 += Math.signum(a1) == Math.signum(a2) ? 1 : 0;

          // three
          a1 = a[i + 2];
          a2 = b[i + 2];
          accA3 += Math.abs(a1);
          accB3 += Math.abs(a2);
          accR3 += Math.signum(a1) == Math.signum(a2) ? 1 : 0;

          // four
          a1 = a[i + 3];
          a2 = b[i + 3];
          accA4 += Math.abs(a1);
          accB4 += Math.abs(a2);
          accR4 += Math.signum(a1) == Math.signum(a2) ? 1 : 0;
        }
        normA += accA1 + accA2 + accA3 + accA4;
        normB += accB1 + accB2 + accB3 + accB4;
        res += accR1 + accR2 + accR3 + accR4;
      }

      for (; i < a.length; i++) {
        a1 = a[i];
        a2 = b[i];
        normA += Math.abs(a1);
        normB += Math.abs(a2);
        res = fma(Math.signum(a1), Math.signum(a2), res);
      }

      return res * (normA / a.length) * (normB / a.length);
    }
  }

  /**
   * Calculates the dot product as the mean of upper and lower bounds.
   * The upper bound is calculated as ||v1|| * ||v2||.
   * The lower bound is calculated  as abs(||v1|| - ||v2||).
   */
  private class BoundMeanApproximateDotProductFunction
          implements FloatApproximateDotProductFunction {

    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      int i = 0;
      float normA = 0;
      float normB = 0;

      // if the array is big, unroll it
      if (a.length > 32) {
        float accA1 = 0;
        float accB1 = 0;

        float accA2 = 0;
        float accB2 = 0;

        float accA3 = 0;
        float accB3 = 0;

        float accA4 = 0;
        float accB4 = 0;

        int upperBound = a.length & ~(4 - 1);
        for (; i < upperBound; i += 4) {
          // one
          accA1 = fma(a[i], a[i], accA1);
          accB1 = fma(b[i], b[i], accB1);

          // two
          accA2 = fma(a[i + 1], a[i + 1], accA2);
          accB2 = fma(b[i + 1], b[i + 1], accB2);

          // three
          accA3 = fma(a[i + 2], a[i + 2], accA3);
          accB3 = fma(b[i + 2], b[i + 2], accB3);

          // four
          accA4 = fma(a[i + 3], a[i + 3], accA4);
          accB4 = fma(b[i + 3], b[i + 3], accB4);
        }
        normA = accA1 + accA2 + accA3 + accA4;
        normB = accB1 + accB2 + accB3 + accB4;
      }

      for (; i < a.length; i++) {
        normA = fma(a[i], a[i], normA);
        normB = fma(b[i], b[i], normB);
      }

      double fna = Math.sqrt(normB);
      double fnb = Math.sqrt(normA);
      return (float) (fna * fnb - Math.abs(fna - fnb)) * 0.5f;
    }
  }

  private static float approxSquareRoot(double d) {
    return (float)
        Double.longBitsToDouble(((Double.doubleToLongBits(d) - (1L << 52)) >> 1) + (1L << 61));
  }

  private class CompositeApprox implements FloatApproximateDotProductFunction {
    private final List<FloatApproximateDotProductFunction> functions;

    public CompositeApprox() {
      this.functions = List.of(
              new MeanSumApproximateDotProductFunction(),
              new Norm2BoundApproximateDotProductFunction(),
              new Norm1BoundApproximateDotProductFunction(),
              new BoundMeanApproximateDotProductFunction(),
              new XorNetApproximateDotProductFunction(),
              new ApproximateXorNetApproximateDotProductFunction(),
              new VectorQuantizationApproximateDotProductFunction(),
              new RandomProjectionsApproximateDotProductFunction());
    }
    @Override
    public float approximateDotProduct(float[] a, float[] b) {
      StringBuilder result = new StringBuilder();
      float f = new DefaultVectorUtilSupport().dotProduct(a, b);
      result.append(f).append(',');
      for (FloatApproximateDotProductFunction c : functions) {
        result.append(c.approximateDotProduct(a, b)).append(',');
      }
      result.append('\n');
      System.err.println(result);
      return f;
    }
  }
}
