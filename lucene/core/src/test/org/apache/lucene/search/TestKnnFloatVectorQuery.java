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

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;
import static org.apache.lucene.index.VectorSimilarityFunction.DOT_PRODUCT;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.QueryTimeout;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.TestVectorUtil;
import org.apache.lucene.util.VectorUtil;

public class TestKnnFloatVectorQuery extends BaseKnnVectorQueryTestCase {

  @Override
  KnnFloatVectorQuery getKnnVectorQuery(String field, float[] query, int k, Query queryFilter) {
    return new KnnFloatVectorQuery(field, query, k, queryFilter);
  }

  @Override
  AbstractKnnVectorQuery getThrowingKnnVectorQuery(String field, float[] vec, int k, Query query) {
    return new ThrowingKnnVectorQuery(field, vec, k, query);
  }

  @Override
  AbstractKnnVectorQuery getCappedResultsThrowingKnnVectorQuery(
      String field, float[] vec, int k, Query query, int maxResults) {
    return new CappedResultsThrowingKnnVectorQuery(field, vec, k, query, maxResults);
  }

  @Override
  float[] randomVector(int dim) {
    return TestVectorUtil.randomVector(dim);
  }

  @Override
  Field getKnnVectorField(
      String name, float[] vector, VectorSimilarityFunction similarityFunction) {
    return new KnnFloatVectorField(name, vector, similarityFunction);
  }

  @Override
  Field getKnnVectorField(String name, float[] vector) {
    return new KnnFloatVectorField(name, vector);
  }

  public void testToString() throws IOException {
    try (Directory indexStore =
            getIndexStore("field", new float[] {0, 1}, new float[] {1, 2}, new float[] {0, 0});
        IndexReader reader = DirectoryReader.open(indexStore)) {
      AbstractKnnVectorQuery query = getKnnVectorQuery("field", new float[] {0.0f, 1.0f}, 10);
      assertEquals("KnnFloatVectorQuery:field[0.0,...][10]", query.toString("ignored"));

      assertDocScoreQueryToString(query.rewrite(newSearcher(reader)));

      // test with filter
      Query filter = new TermQuery(new Term("id", "text"));
      query = getKnnVectorQuery("field", new float[] {0.0f, 1.0f}, 10, filter);
      assertEquals("KnnFloatVectorQuery:field[0.0,...][10][id:text]", query.toString("ignored"));
    }
  }

  public void testVectorEncodingMismatch() throws IOException {
    try (Directory indexStore =
            getIndexStore("field", new float[] {0, 1}, new float[] {1, 2}, new float[] {0, 0});
        IndexReader reader = DirectoryReader.open(indexStore)) {
      Query filter = null;
      if (random().nextBoolean()) {
        filter = new MatchAllDocsQuery();
      }
      AbstractKnnVectorQuery query = new KnnByteVectorQuery("field", new byte[] {0, 1}, 10, filter);
      IndexSearcher searcher = newSearcher(reader);
      expectThrows(IllegalStateException.class, () -> searcher.search(query, 10));
    }
  }

  public void testGetTarget() {
    float[] queryVector = new float[] {0, 1};
    KnnFloatVectorQuery q1 = new KnnFloatVectorQuery("f1", queryVector, 10);

    assertArrayEquals(queryVector, q1.getTargetCopy(), 0);
    assertNotEquals(queryVector, q1.getTargetCopy());
  }

  public void testScoreNegativeDotProduct() throws IOException {
    try (Directory d = newDirectory()) {
      try (IndexWriter w = new IndexWriter(d, new IndexWriterConfig())) {
        Document doc = new Document();
        doc.add(getKnnVectorField("field", new float[] {-1, 0}, DOT_PRODUCT));
        w.addDocument(doc);
        doc = new Document();
        doc.add(getKnnVectorField("field", new float[] {1, 0}, DOT_PRODUCT));
        w.addDocument(doc);
      }
      try (IndexReader reader = DirectoryReader.open(d)) {
        assertEquals(1, reader.leaves().size());
        IndexSearcher searcher = new IndexSearcher(reader);
        AbstractKnnVectorQuery query = getKnnVectorQuery("field", new float[] {1, 0}, 2);
        Query rewritten = query.rewrite(searcher);
        Weight weight = searcher.createWeight(rewritten, ScoreMode.COMPLETE, 1);
        Scorer scorer = weight.scorer(reader.leaves().get(0));

        // scores are normalized to lie in [0, 1]
        DocIdSetIterator it = scorer.iterator();
        assertEquals(2, it.cost());
        assertEquals(0, it.nextDoc());
        assertTrue(0 <= scorer.score());
        assertEquals(1, it.advance(1));
        assertEquals(1, scorer.score(), EPSILON);
      }
    }
  }

  public void testScoreDotProduct() throws IOException {
    try (Directory d = newDirectory()) {
      try (IndexWriter w = new IndexWriter(d, configStandardCodec())) {
        for (int j = 1; j <= 5; j++) {
          Document doc = new Document();
          doc.add(
              getKnnVectorField(
                  "field", VectorUtil.l2normalize(new float[] {j, j * j}), DOT_PRODUCT));
          w.addDocument(doc);
        }
      }
      try (IndexReader reader = DirectoryReader.open(d)) {
        assertEquals(1, reader.leaves().size());
        IndexSearcher searcher = new IndexSearcher(reader);
        AbstractKnnVectorQuery query =
            getKnnVectorQuery("field", VectorUtil.l2normalize(new float[] {2, 3}), 3);
        Query rewritten = query.rewrite(searcher);
        Weight weight = searcher.createWeight(rewritten, ScoreMode.COMPLETE, 1);
        Scorer scorer = weight.scorer(reader.leaves().get(0));

        // prior to advancing, score is undefined
        assertEquals(-1, scorer.docID());
        expectThrows(ArrayIndexOutOfBoundsException.class, scorer::score);

        /* score0 = ((2,3) * (1, 1) = 5) / (||2, 3|| * ||1, 1|| = sqrt(26)), then
         * normalized by (1 + x) /2.
         */
        float score0 =
            (float) ((1 + (2 * 1 + 3 * 1) / Math.sqrt((2 * 2 + 3 * 3) * (1 * 1 + 1 * 1))) / 2);

        /* score1 = ((2,3) * (2, 4) = 16) / (||2, 3|| * ||2, 4|| = sqrt(260)), then
         * normalized by (1 + x) /2
         */
        float score1 =
            (float) ((1 + (2 * 2 + 3 * 4) / Math.sqrt((2 * 2 + 3 * 3) * (2 * 2 + 4 * 4))) / 2);

        // doc 1 happens to have the max score
        assertEquals(score1, scorer.getMaxScore(2), 0.001);
        assertEquals(score1, scorer.getMaxScore(Integer.MAX_VALUE), 0.0001);

        DocIdSetIterator it = scorer.iterator();
        assertEquals(3, it.cost());
        assertEquals(0, it.nextDoc());
        // doc 0 has (1, 1)
        assertEquals(score0, scorer.score(), 0.0001);
        assertEquals(1, it.advance(1));
        assertEquals(score1, scorer.score(), 0.0001);

        // since topK was 3
        assertEquals(NO_MORE_DOCS, it.advance(4));
        expectThrows(ArrayIndexOutOfBoundsException.class, scorer::score);
      }
    }
  }

  public void testDocAndScoreQueryBasics() throws IOException {
    try (Directory directory = newDirectory()) {
      final DirectoryReader reader;
      try (RandomIndexWriter iw = new RandomIndexWriter(random(), directory)) {
        for (int i = 0; i < 50; i++) {
          Document doc = new Document();
          doc.add(new StringField("field", "value" + i, Field.Store.NO));
          iw.addDocument(doc);
          if (i % 10 == 0) {
            iw.flush();
          }
        }
        reader = iw.getReader();
      }
      try (reader) {
        IndexSearcher searcher = LuceneTestCase.newSearcher(reader);
        List<ScoreDoc> scoreDocsList = new ArrayList<>();
        for (int doc = 0; doc < 30; doc += 1 + random().nextInt(5)) {
          scoreDocsList.add(new ScoreDoc(doc, randomFloat()));
        }
        ScoreDoc[] scoreDocs = scoreDocsList.toArray(new ScoreDoc[0]);
        int[] docs = new int[scoreDocs.length];
        float[] scores = new float[scoreDocs.length];
        float maxScore = Float.MIN_VALUE;
        for (int i = 0; i < scoreDocs.length; i++) {
          docs[i] = scoreDocs[i].doc;
          scores[i] = scoreDocs[i].score;
          maxScore = Math.max(maxScore, scores[i]);
        }
        IndexReader indexReader = searcher.getIndexReader();
        int[] segments = DocAndScoreQuery.findSegmentStarts(indexReader.leaves(), docs);

        DocAndScoreQuery query =
            new DocAndScoreQuery(
                docs,
                scores,
                maxScore,
                segments,
                scoreDocs.length,
                indexReader.getContext().id(),
                0);
        final Weight w = query.createWeight(searcher, ScoreMode.TOP_SCORES, 1.0f);
        TopDocs topDocs = searcher.search(query, 100);
        assertEquals(scoreDocs.length, topDocs.totalHits.value());
        assertEquals(query.visited(), topDocs.totalHits.value());
        assertEquals(TotalHits.Relation.EQUAL_TO, topDocs.totalHits.relation());
        Arrays.sort(topDocs.scoreDocs, Comparator.comparingInt(scoreDoc -> scoreDoc.doc));
        assertEquals(scoreDocs.length, topDocs.scoreDocs.length);
        assertEquals(0, query.reentryCount());
        for (int i = 0; i < scoreDocs.length; i++) {
          assertEquals(scoreDocs[i].doc, topDocs.scoreDocs[i].doc);
          assertEquals(scoreDocs[i].score, topDocs.scoreDocs[i].score, 0.0001f);
          assertTrue(searcher.explain(query, scoreDocs[i].doc).isMatch());
        }

        for (LeafReaderContext leafReaderContext : searcher.getLeafContexts()) {
          final Scorer scorer = w.scorer(leafReaderContext);
          final int count = w.count(leafReaderContext);
          if (scorer == null) {
            assertEquals(0, count);
          } else {
            assertTrue(leafReaderContext.toString(), scorer.getMaxScore(NO_MORE_DOCS) > 0.0f);
            assertTrue(count > 0);
            int iteratorCount = 0;
            while (scorer.iterator().nextDoc() != NO_MORE_DOCS) {
              iteratorCount++;
            }
            assertEquals(iteratorCount, count);
          }
        }
      }
    }
  }

  static class ThrowingKnnVectorQuery extends KnnFloatVectorQuery {

    public ThrowingKnnVectorQuery(String field, float[] target, int k, Query filter) {
      super(field, target, k, filter, new KnnSearchStrategy.Hnsw(0));
    }

    @Override
    protected TopDocs exactSearch(
        LeafReaderContext context, DocIdSetIterator acceptIterator, QueryTimeout queryTimeout) {
      throw new UnsupportedOperationException("exact search is not supported");
    }

    @Override
    public String toString(String field) {
      return null;
    }
  }

  static class CappedResultsThrowingKnnVectorQuery extends ThrowingKnnVectorQuery {

    private final int maxResults;

    public CappedResultsThrowingKnnVectorQuery(
        String field, float[] target, int k, Query filter, int maxResults) {
      super(field, target, k, filter);
      this.maxResults = maxResults;
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        Bits acceptDocs,
        int visitedLimit,
        KnnCollectorManager knnCollectorManager)
        throws IOException {
      TopDocs topDocs =
          super.approximateSearch(context, acceptDocs, Integer.MAX_VALUE, knnCollectorManager);
      long results = Math.min(topDocs.totalHits.value(), maxResults);
      ScoreDoc[] scoreDocs = new ScoreDoc[(int) results];
      System.arraycopy(topDocs.scoreDocs, 0, scoreDocs, 0, scoreDocs.length);

      return new TopDocs(new TotalHits(results, TotalHits.Relation.EQUAL_TO), scoreDocs);
    }
  }
}
