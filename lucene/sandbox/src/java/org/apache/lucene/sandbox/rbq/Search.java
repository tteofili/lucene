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

package org.apache.lucene.sandbox.rbq;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.SuppressForbidden;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

/** Class for testing binary quantization */
public class Search {

  // FIXME: Temporary to help with debugging and iteration
  @SuppressForbidden(reason = "Used for testing")
  public static void main(String[] args) throws Exception {
    String dataset = args[2];
    Path basePath = Paths.get(args[0]);
    Path dataVecPath = Paths.get(basePath.toString(), dataset + "_base.fvecs");
    Path queryVecPath = Paths.get(basePath.toString(), dataset + "_query.fvecs");
    int numDocs = 100;
    int dim = Integer.parseInt(args[4]);
    int k = Integer.parseInt(args[5]);
    int totalQueryVectors = Integer.parseInt(args[6]);
    int numDataVectors = Integer.parseInt(args[7]);
    VectorSimilarityFunction vectorSimilarity =
        "eucl".equals(args[8])
            ? VectorSimilarityFunction.EUCLIDEAN
            : VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;
    Path groundTruthPath = Paths.get(basePath.toString(), dataset + "_groundtruth.ivecs");

    ExecutorService executorService = null;
    Path indexPath = Paths.get(args[1]);
    Directory directory = FSDirectory.open(indexPath);
    DirectoryReader reader = DirectoryReader.open(directory);
    IndexSearcher searcher = new IndexSearcher(reader, executorService);
    searcher.setQueryCache(null); // don't bench the cache
    searcher.setSimilarity(IndexSearcher.getDefaultSimilarity());

    System.out.println(
        "Searcher: numDocs="
            + searcher.getIndexReader().numDocs()
            + " maxDoc="
            + searcher.getIndexReader().maxDoc());
    IndexSearcher.LeafSlice[] slices =
        IndexSearcher.slices(searcher.getIndexReader().leaves(), 250_000, 5, false);
    System.out.println(
        "Reader has "
            + slices.length
            + " slices, from "
            + searcher.getIndexReader().leaves().size()
            + " segments:");

    int expandedVectorSize = 0;
    long totalTime = 0;
    int correctCount = 0;
    try (MMapDirectory queryDirectory = new MMapDirectory(basePath);
        IndexInput queryVectorInput =
            queryDirectory.openInput(queryVecPath.toString(), IOContext.DEFAULT);
        IndexInput dataVectorInput =
            queryDirectory.openInput(dataVecPath.toString(), IOContext.DEFAULT)) {
      int[][] G = readGroundTruth(groundTruthPath.toString(), queryDirectory, totalQueryVectors);
      RandomAccessVectorValues.Floats vectorValues =
          new VectorsReaderWithOffset(queryVectorInput, totalQueryVectors, dim, Integer.BYTES);
      RandomAccessVectorValues.Floats dataVectors =
          new VectorsReaderWithOffset(dataVectorInput, numDataVectors, dim, Integer.BYTES);
      long startTime = System.nanoTime();
      for (int i = 0; i < totalQueryVectors; i++) {
        float[] queryVector = vectorValues.vectorValue(i);
        if (expandedVectorSize > 0) {
          float[] expansion = new float[expandedVectorSize];
          Arrays.fill(expansion, 0.9f);
          int vectorSize = queryVector.length;
          queryVector = Arrays.copyOf(queryVector, vectorSize + expansion.length);
          System.arraycopy(expansion, 0, queryVector, vectorSize, expansion.length);
        }
        Query q = new KnnFloatVectorQuery("knn", queryVector, numDocs);
        TopDocs collectedDocs = searcher.search(q, numDocs);
        long nextTime = System.nanoTime();
        totalTime += nextTime - startTime;
        startTime = nextTime;

        HitQueue KNNs = new HitQueue(k, false);
        // rescore & get top k
        for (int j = 0; j < collectedDocs.scoreDocs.length; j++) {
          Document doc = searcher.storedFields().document(collectedDocs.scoreDocs[j].doc);
          int id = Integer.parseInt(doc.get("id"));
          float[] dataVector = dataVectors.vectorValue(id);
          if (expandedVectorSize > 0) {
            float[] expansion = new float[expandedVectorSize];
            Arrays.fill(expansion, 0.9f);
            int dataVectorSize = dataVector.length;
            dataVector = Arrays.copyOf(dataVector, dataVectorSize + expansion.length);
            System.arraycopy(expansion, 0, dataVector, dataVectorSize, expansion.length);
          }
          float rawScore = vectorSimilarity.compare(dataVector, queryVector);
          KNNs.insertWithOverflow(new ScoreDoc(id, rawScore));
        }

        int correct = 0;
        while (KNNs.size() > 0) {
          int id = KNNs.pop().doc;
          for (int j = 0; j < k; j++) {
            if (id == G[i][j]) {
              correct++;
            }
          }
        }
        correctCount += correct;
      }
      System.out.println("Total Time (ms): " + (totalTime / 1000000));
    }

    float recall = (float) correctCount / (totalQueryVectors * k);
    System.out.println("Recall: " + recall);
  }

  private static int[][] readGroundTruth(
      String groundTruthFile, FSDirectory directory, int numQueries) throws IOException {
    if (!Files.exists(directory.getDirectory().resolve(groundTruthFile))) {
      return null;
    }
    int[][] groundTruths = new int[numQueries][];
    // reading the ground truths from the file
    try (IndexInput queryGroundTruthInput =
        directory.openInput(groundTruthFile, IOContext.DEFAULT)) {
      for (int i = 0; i < numQueries; i++) {
        int length = queryGroundTruthInput.readInt();
        groundTruths[i] = new int[length];
        for (int j = 0; j < length; j++) {
          groundTruths[i][j] = queryGroundTruthInput.readInt();
        }
      }
    }
    return groundTruths;
  }
}