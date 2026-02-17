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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene90.IndexedDISI;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TimeLimitingKnnCollectorManager;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.search.knn.TopKnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.apache.lucene.search.TotalHits;

/**
 * Uses {@link KnnVectorsReader#search} to perform nearest neighbour search.
 *
 * <p>This query also allows for performing a kNN search subject to a filter. In this case, it first
 * executes the filter for each leaf, then chooses a strategy dynamically:
 *
 * <ul>
 *   <li>If filter cost is less than k, just execute an exact search
 *   <li>Otherwise run a kNN search subject to filter
 *   <li>If kNN search visits too many vectors without completing, stop and run an exact search
 * </ul>
 */
public abstract class AbstractKnnVectorQuery extends Query {

  protected final String field;
  protected final int k;
  protected final Query filter;
  protected final KnnSearchStrategy searchStrategy;
  protected final VectorSlice vectorSlice;
  
  public String getField() {
    return field;
  }
  
  public int getK() {
    return k;
  }
  
  public Query getFilter() {
    return filter;
  }
  
  public VectorSlice getVectorSlice() {
    return vectorSlice;
  }

  AbstractKnnVectorQuery(String field, int k, Query filter, KnnSearchStrategy searchStrategy) {
    this(field, k, filter, searchStrategy, VectorSlice.NONE);
  }

  AbstractKnnVectorQuery(String field, int k, Query filter, KnnSearchStrategy searchStrategy, VectorSlice vectorSlice) {
    this.field = Objects.requireNonNull(field, "field");
    this.k = k;
    if (k < 1) {
      throw new IllegalArgumentException("k must be at least 1, got: " + k);
    }
    this.filter = filter;
    this.searchStrategy = searchStrategy;
    this.vectorSlice = Objects.requireNonNull(vectorSlice, "vectorSlice");
  }

  // Simplified weight creation - to be implemented by subclasses

  /**
   * Returns the string representation of the target vector.
   */
  protected abstract String targetString();
  
  
  
  /**
   * Creates a vector scorer for exact search.
   */
  protected abstract VectorScorer createVectorScorer(LeafReaderContext context, FieldInfo fi) throws IOException;

  @Override
  public String toString(String field) {
    StringBuilder buffer = new StringBuilder();
    buffer.append(getClass().getSimpleName() + ":");
    buffer.append(this.field + "[" + targetString() + ",...]");
    buffer.append("[" + k + "]");
    if (!vectorSlice.isNoSlice()) {
      buffer.append("[slice:" + vectorSlice + "]");
    }
    if (this.filter != null) {
      buffer.append("[" + this.filter + "]");
    }
    return buffer.toString();
  }

  

  /**
   * Creates an effective reader with dimension translation if needed.
   */
  protected LeafReader getEffectiveReader(LeafReader originalReader) {
    // For now, just return original reader
    // Dimension translation will be applied during vector access
    return originalReader;
  }

  // Legacy method signature for backward compatibility
  @Deprecated
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      AcceptDocs acceptDocs,
      int visitedLimit,
      KnnCollectorManager knnCollectorManager)
      throws IOException {
    // Convert to new signature for implementations that override this method
    Weight filterWeight = null;
    TimeLimitingKnnCollectorManager timeLimitingKnnCollectorManager = 
        knnCollectorManager != null ? new TimeLimitingKnnCollectorManager(knnCollectorManager, null) : null;
    return approximateSearch(context, filterWeight, timeLimitingKnnCollectorManager);
  }
  
  // Current method signature for new implementations
  protected abstract TopDocs approximateSearch(
      LeafReaderContext context,
      Weight filterWeight,
      TimeLimitingKnnCollectorManager timeLimitingKnnCollectorManager)
      throws IOException;

  // Simplified implementation - removed complex collection logic

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    AbstractKnnVectorQuery that = (AbstractKnnVectorQuery) o;
    return k == that.k
        && Objects.equals(field, that.field)
        && Objects.equals(filter, that.filter)
        && Objects.equals(searchStrategy, that.searchStrategy)
        && Objects.equals(vectorSlice, that.vectorSlice);
  }

  @Override
  public int hashCode() {
    return Objects.hash(k, field, filter, searchStrategy, vectorSlice);
  }

  protected static final TopDocs NO_RESULTS = new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new ScoreDoc[0]);
}