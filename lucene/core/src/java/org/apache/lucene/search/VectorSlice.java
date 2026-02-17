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
package org.apache.lucene.search;

import java.util.Objects;

/**
 * Configuration for vector slicing/truncation at runtime.
 * 
 * <p>This class allows slicing vectors during query execution without modifying
 * the indexed vectors. The slice is applied to both the query vector and the 
 * document vectors before similarity computation.
 *
 * @lucene.experimental
 */
public class VectorSlice {
  
  /** No slicing - use the full vector */
  public static final VectorSlice NONE = new VectorSlice(0, Integer.MAX_VALUE);
  
  private final int startIndex;
  private final int endIndex;
  
  /**
   * Creates a vector slice configuration.
   * 
   * @param startIndex the inclusive start index of the slice (0-based)
   * @param endIndex the exclusive end index of the slice
   * @throws IllegalArgumentException if indices are invalid
   */
  public VectorSlice(int startIndex, int endIndex) {
    if (startIndex < 0) {
      throw new IllegalArgumentException("startIndex must be >= 0, got: " + startIndex);
    }
    if (endIndex <= startIndex) {
      throw new IllegalArgumentException("endIndex must be > startIndex, got startIndex: " + startIndex + ", endIndex: " + endIndex);
    }
    this.startIndex = startIndex;
    this.endIndex = endIndex;
  }
  
  /**
   * Creates a slice from the beginning up to the specified length.
   * 
   * @param length the length of the slice (must be > 0)
   * @return a slice from index 0 to length
   */
  public static VectorSlice fromLength(int length) {
    return new VectorSlice(0, length);
  }
  
  /**
   * Creates a slice from the start index to the end of the vector.
   * 
   * @param startIndex the inclusive start index
   * @return a slice from startIndex to the end
   */
  public static VectorSlice fromStart(int startIndex) {
    return new VectorSlice(startIndex, Integer.MAX_VALUE);
  }
  
  /**
   * Returns the length of this slice when applied to a vector of the given dimension.
   * 
   * @param vectorDimension the dimension of the vector
   * @return the slice length
   */
  public int getSliceLength(int vectorDimension) {
    return Math.min(endIndex, vectorDimension) - Math.min(startIndex, vectorDimension);
  }
  
  /**
   * Applies this slice to the given vector.
   * 
   * @param vector the original vector
   * @return the sliced vector
   * @throws IllegalArgumentException if slice bounds exceed vector dimensions
   */
  public float[] apply(float[] vector) {
    int actualStart = Math.min(startIndex, vector.length);
    int actualEnd = Math.min(endIndex, vector.length);
    
    if (actualStart >= actualEnd) {
      throw new IllegalArgumentException("Slice bounds [" + startIndex + ", " + endIndex + 
                                         ") exceed vector dimension " + vector.length);
    }
    
    float[] result = new float[actualEnd - actualStart];
    System.arraycopy(vector, actualStart, result, 0, result.length);
    return result;
  }
  
  /**
   * Applies this slice to the given byte vector.
   * 
   * @param vector the original vector
   * @return the sliced vector
   * @throws IllegalArgumentException if slice bounds exceed vector dimensions
   */
  public byte[] apply(byte[] vector) {
    int actualStart = Math.min(startIndex, vector.length);
    int actualEnd = Math.min(endIndex, vector.length);
    
    if (actualStart >= actualEnd) {
      throw new IllegalArgumentException("Slice bounds [" + startIndex + ", " + endIndex + 
                                         ") exceed vector dimension " + vector.length);
    }
    
    byte[] result = new byte[actualEnd - actualStart];
    System.arraycopy(vector, actualStart, result, 0, result.length);
    return result;
  }
  
  /**
   * Returns true if this slice represents the full vector (no slicing).
   */
  public boolean isNoSlice() {
    return startIndex == 0 && endIndex == Integer.MAX_VALUE;
  }
  
  /**
   * Returns the inclusive start index.
   */
  public int getStartIndex() {
    return startIndex;
  }
  
  /**
   * Returns the exclusive end index.
   */
  public int getEndIndex() {
    return endIndex;
  }
  
  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    VectorSlice that = (VectorSlice) o;
    return startIndex == that.startIndex && endIndex == that.endIndex;
  }
  
  @Override
  public int hashCode() {
    return Objects.hash(startIndex, endIndex);
  }
  
  @Override
  public String toString() {
    if (isNoSlice()) {
      return "VectorSlice.NONE";
    }
    return "VectorSlice[" + startIndex + ", " + endIndex + "]";
  }
}