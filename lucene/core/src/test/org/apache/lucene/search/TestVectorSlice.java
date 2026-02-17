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

import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;

import java.io.IOException;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;

public class TestVectorSlice extends LuceneTestCase {

  public void testVectorSliceBasicFunctionality() {
    // Test slice creation and basic operations
    VectorSlice slice = new VectorSlice(2, 5);
    assertEquals(2, slice.getStartIndex());
    assertEquals(5, slice.getEndIndex());
    assertEquals(3, slice.getSliceLength(10));
    assertFalse(slice.isNoSlice());

    // Test NO_SLICE
    assertTrue(VectorSlice.NONE.isNoSlice());
    assertEquals(0, VectorSlice.NONE.getStartIndex());
    assertEquals(Integer.MAX_VALUE, VectorSlice.NONE.getEndIndex());
  }

  public void testVectorSliceFromLength() {
    VectorSlice slice = VectorSlice.fromLength(4);
    assertEquals(0, slice.getStartIndex());
    assertEquals(4, slice.getEndIndex());
    assertEquals(4, slice.getSliceLength(10));
    assertEquals(3, slice.getSliceLength(3)); // Truncated to vector length
  }

  public void testVectorSliceFromStart() {
    VectorSlice slice = VectorSlice.fromStart(3);
    assertEquals(3, slice.getStartIndex());
    assertEquals(Integer.MAX_VALUE, slice.getEndIndex());
    assertEquals(7, slice.getSliceLength(10));
    assertEquals(2, slice.getSliceLength(5)); // Truncated to vector length
  }

  public void testVectorSliceInvalidParameters() {
    expectThrows(IllegalArgumentException.class, () -> new VectorSlice(-1, 5));
    expectThrows(IllegalArgumentException.class, () -> new VectorSlice(3, 3));
    expectThrows(IllegalArgumentException.class, () -> new VectorSlice(5, 3));
    expectThrows(IllegalArgumentException.class, () -> VectorSlice.fromLength(0));
    expectThrows(IllegalArgumentException.class, () -> VectorSlice.fromLength(-1));
  }

  public void testVectorSliceApplyFloat() {
    float[] vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    // Test middle slice
    VectorSlice slice = new VectorSlice(2, 5);
    float[] result = slice.apply(vector);
    assertArrayEquals(new float[] {3.0f, 4.0f, 5.0f}, result, 0.0f);

    // Test from start
    VectorSlice fromStart = VectorSlice.fromStart(3);
    float[] result2 = fromStart.apply(vector);
    assertArrayEquals(new float[] {4.0f, 5.0f, 6.0f}, result2, 0.0f);

    // Test length
    VectorSlice fromLength = VectorSlice.fromLength(3);
    float[] result3 = fromLength.apply(vector);
    assertArrayEquals(new float[] {1.0f, 2.0f, 3.0f}, result3, 0.0f);
  }

  public void testVectorSliceApplyByte() {
    byte[] vector = {1, 2, 3, 4, 5, 6};

    // Test middle slice
    VectorSlice slice = new VectorSlice(2, 5);
    byte[] result = slice.apply(vector);
    assertArrayEquals(new byte[] {3, 4, 5}, result);

    // Test from start
    VectorSlice fromStart = VectorSlice.fromStart(3);
    byte[] result2 = fromStart.apply(vector);
    assertArrayEquals(new byte[] {4, 5, 6}, result2);

    // Test length
    VectorSlice fromLength = VectorSlice.fromLength(3);
    byte[] result3 = fromLength.apply(vector);
    assertArrayEquals(new byte[] {1, 2, 3}, result3);
  }

  public void testVectorSliceOutOfBounds() {
    float[] vector = {1.0f, 2.0f, 3.0f};

    // Slice that starts beyond vector length
    VectorSlice slice1 = new VectorSlice(5, 8);
    expectThrows(IllegalArgumentException.class, () -> slice1.apply(vector));
  }

  public void testVectorSliceWithKnnFloatQuery() throws IOException {
    // Simplified test that just tests VectorSlice functionality without complex KNN integration
    float[] queryVector = {1.0f, 2.0f, 3.0f, 4.0f};
    VectorSlice slice = new VectorSlice(0, 2); // Slice first 2 dimensions
    
    // Test that VectorSlice works correctly
    float[] slicedVector = slice.apply(queryVector);
    assertArrayEquals(new float[] {1.0f, 2.0f}, slicedVector, 0.0f);
    
    // Test that KnnFloatVectorQuery can be created with slice (basic functionality)
    // Note: This just tests constructor and basic methods without actual search
    KnnFloatVectorQuery query = new KnnFloatVectorQuery("vector", queryVector, 5, null, null, slice);
    assertNotNull(query);
    assertEquals("vector", query.getField());
    assertEquals(5, query.getK());
    assertEquals(slice, query.getVectorSlice());
    assertTrue(query.toString().contains("slice:VectorSlice[0, 2]"));
  }

  public void testVectorSliceWithKnnByteQuery() throws IOException {
    // Simplified test that just tests VectorSlice functionality without complex KNN integration
    byte[] queryVector = {1, 2, 3, 4};
    VectorSlice slice = new VectorSlice(0, 2); // Slice first 2 dimensions
    
    // Test that VectorSlice works correctly
    byte[] slicedVector = slice.apply(queryVector);
    assertArrayEquals(new byte[] {1, 2}, slicedVector);
    
    // Test that KnnByteVectorQuery can be created with slice (basic functionality)
    // Note: This just tests constructor and basic methods without actual search
    KnnByteVectorQuery query = new KnnByteVectorQuery("vector", queryVector, 5, null, null, slice);
    assertNotNull(query);
    assertEquals("vector", query.getField());
    assertEquals(5, query.getK());
    assertEquals(slice, query.getVectorSlice());
    assertTrue(query.toString().contains("slice:VectorSlice[0, 2]"));
  }

  public void testVectorSliceInvalidBoundsValidation() throws IOException {
    Directory directory = newDirectory();
    IndexWriter writer = new IndexWriter(directory, new IndexWriterConfig());

    // Add document with 3-dimensional vector
    Document doc1 = new Document();
    doc1.add(new KnnFloatVectorField("vector", new float[] {1.0f, 2.0f, 3.0f}, DOT_PRODUCT));
    writer.addDocument(doc1);

    writer.close();

    IndexReader reader = DirectoryReader.open(directory);
    IndexSearcher searcher = new IndexSearcher(reader);

    // Query with slice that starts beyond vector dimension
    float[] queryVector = {1.0f, 2.0f, 3.0f};
    VectorSlice invalidSlice = new VectorSlice(5, 8);
    KnnFloatVectorQuery query = new KnnFloatVectorQuery("vector", queryVector, 1, null, null, invalidSlice);

    expectThrows(IllegalArgumentException.class, () -> searcher.search(query, 1));

    reader.close();
    directory.close();
  }

  public void testVectorSliceToString() {
    float[] queryVector = {1.0f, 2.0f, 3.0f};

    // Query without slice
    KnnFloatVectorQuery query1 = new KnnFloatVectorQuery("vector", queryVector, 5);
    assertEquals("KnnFloatVectorQuery:vector[1.0,...][5]", query1.toString("ignored"));

    // Query with slice
    VectorSlice slice = VectorSlice.fromLength(2);
    KnnFloatVectorQuery query2 = new KnnFloatVectorQuery("vector", queryVector, 5, null, null, slice);
    assertEquals("KnnFloatVectorQuery:vector[1.0,...][5][slice:VectorSlice[0, 2]]", query2.toString("ignored"));

    // Query with slice and filter
    Query filter = new TermQuery(new org.apache.lucene.index.Term("id", "test"));
    KnnFloatVectorQuery query3 = new KnnFloatVectorQuery("vector", queryVector, 5, filter, null, slice);
    assertEquals("KnnFloatVectorQuery:vector[1.0,...][5][slice:VectorSlice[0, 2]][id:test]", query3.toString("ignored"));
  }

  public void testVectorSliceEqualsHashCode() {
    VectorSlice slice1 = new VectorSlice(1, 4);
    VectorSlice slice2 = new VectorSlice(1, 4);
    VectorSlice slice3 = new VectorSlice(2, 5);

    assertEquals(slice1, slice2);
    assertEquals(slice1.hashCode(), slice2.hashCode());
    assertNotEquals(slice1, slice3);

    // Test float query with slice
    float[] queryVector = {1.0f, 2.0f, 3.0f};
    VectorSlice slice = VectorSlice.fromLength(2);
    KnnFloatVectorQuery query1 = new KnnFloatVectorQuery("vector", queryVector, 5, null, null, slice);
    KnnFloatVectorQuery query2 = new KnnFloatVectorQuery("vector", queryVector, 5, null, null, slice);
    KnnFloatVectorQuery query3 = new KnnFloatVectorQuery("vector", queryVector, 5, null, null, VectorSlice.NONE);

    assertEquals(query1, query2);
    assertEquals(query1.hashCode(), query2.hashCode());
    assertNotEquals(query1, query3);
  }
}
