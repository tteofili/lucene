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
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.Bits;

import org.apache.lucene.util.VectorUtil;

/**
 * Query that finds the k nearest float vectors to a target vector and returns those documents. This
 * query will run in exact mode if the filter can execute efficiently with N a few docs.
 * Otherwise it will run in approximate mode using HNSW indexing.
 */
public class KnnFloatVectorQuery extends AbstractKnnVectorQuery {

  private final float[] target;

  /**
   * Find the <code>k</code> nearest documents to the target vector according to the vectors in the
   * given field. <code>target</code> vector.
   *
   * @param field a field that has been indexed as a KNN vector field.
   * @param target the target of the search
   * @param k the number of documents to find
   * @throws IllegalArgumentException if {@code target} has a different dimension than the field or
   *     contains an invalid element (NaN, inf or -inf)
   */
  public KnnFloatVectorQuery(String field, float[] target, int k) {
    this(field, target, k, null);
  }

  /**
   * Find the <code>k</code> nearest documents to the target vector according to the vectors in the
   * given field. <code>target</code> vector.
   *
   * @param field a field that has been indexed as a KNN vector field.
   * @param target the target of the search
   * @param k the number of documents to find
   * @param filter a filter applied before the vector search
   * @throws IllegalArgumentException if {@code target} has a different dimension than the field or
   *     contains an invalid element (NaN, inf or -inf)
   */
  public KnnFloatVectorQuery(String field, float[] target, int k, Query filter) {
    this(field, target, k, filter, null);
  }

  /**
   * Find the <code>k</code> nearest documents to the target vector according to the vectors in the
   * given field. <code>target</code> vector.
   *
   * @param field a field that has been indexed as a KNN vector field.
   * @param target the target of the search
   * @param k the number of documents to find
   * @param filter a filter applied before the vector search
   * @param searchStrategy the search strategy to use. If null, the default strategy will be used.
   *     The underlying format may not support all strategies and is free to ignore the requested
   *     strategy.
   */
  public KnnFloatVectorQuery(String field, float[] target, int k, Query filter, KnnSearchStrategy searchStrategy) {
    this(field, target, k, filter, searchStrategy, VectorSlice.NONE);
  }

  /**
   * Find the <code>k</code> nearest documents to the target vector according to the vectors in the
   * given field. <code>target</code> vector.
   *
   * @param field a field that has been indexed as a KNN vector field.
   * @param target the target of the search
   * @param k the number of documents to find
   * @param filter a filter applied before the vector search
   * @param searchStrategy the search strategy to use. If null, the default strategy will be used.
   *     The underlying format may not support all strategies and is free to ignore the requested
   *     strategy.
   * @param vectorSlice vector slice configuration to apply during search
   * @lucene.experimental
   */
  public KnnFloatVectorQuery(
      String field, float[] target, int k, Query filter, KnnSearchStrategy searchStrategy, VectorSlice vectorSlice) {
    super(field, k, filter, searchStrategy, vectorSlice);
    this.target = VectorUtil.checkFinite(target);
  }

  @Override
  public String targetString() {
    return target.length + " dimensions";
  }

  @Override
  protected TopDocs approximateSearch(
      LeafReaderContext context,
      Weight filterWeight,
      TimeLimitingKnnCollectorManager timeLimitingKnnCollectorManager)
      throws IOException {
    LeafReader reader = getEffectiveReader(context.reader());
    FloatVectorValues floatVectorValues = reader.getFloatVectorValues(field);
    if (floatVectorValues == null) {
      FloatVectorValues.checkField(reader, field);
      return NO_RESULTS;
    }
    if (floatVectorValues.size() == 0) {
      return NO_RESULTS;
    }
    float[] effectiveTarget = target;
    if (!vectorSlice.isNoSlice()) {
      effectiveTarget = vectorSlice.apply(target);
    }
    // Simple implementation using reader.searchNearestVectors
    return reader.searchNearestVectors(field, effectiveTarget, k, null, 0);
  }

  @Override
  protected VectorScorer createVectorScorer(LeafReaderContext context, FieldInfo fi) throws IOException {
    LeafReader reader = getEffectiveReader(context.reader());
    FloatVectorValues vectorValues = reader.getFloatVectorValues(field);
    if (vectorValues == null) {
      FloatVectorValues.checkField(reader, field);
      return null;
    }
    float[] effectiveTarget = target;
    if (!vectorSlice.isNoSlice()) {
      effectiveTarget = vectorSlice.apply(target);
    }
    return vectorValues.scorer(effectiveTarget);
  }

  @Override
  public void visit(QueryVisitor visitor) {
    // Simple implementation - could be enhanced if needed
    if (filter != null) {
      filter.visit(visitor);
    }
  }
}