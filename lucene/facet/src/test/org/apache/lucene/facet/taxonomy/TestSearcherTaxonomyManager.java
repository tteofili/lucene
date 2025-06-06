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
package org.apache.lucene.facet.taxonomy;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.facet.FacetField;
import org.apache.lucene.facet.FacetResult;
import org.apache.lucene.facet.FacetTestCase;
import org.apache.lucene.facet.Facets;
import org.apache.lucene.facet.FacetsCollector;
import org.apache.lucene.facet.FacetsCollectorManager;
import org.apache.lucene.facet.FacetsConfig;
import org.apache.lucene.facet.taxonomy.SearcherTaxonomyManager.SearcherAndTaxonomy;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyReader;
import org.apache.lucene.facet.taxonomy.directory.DirectoryTaxonomyWriter;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexCommit;
import org.apache.lucene.index.IndexNotFoundException;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoDeletionPolicy;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.ReferenceManager;
import org.apache.lucene.search.RefreshCommitSupplier;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.SuppressForbidden;

@LuceneTestCase.SuppressCodecs("SimpleText")
public class TestSearcherTaxonomyManager extends FacetTestCase {

  private static class IndexerThread extends Thread {

    private IndexWriter w;
    private FacetsConfig config;
    private TaxonomyWriter tw;
    private ReferenceManager<SearcherAndTaxonomy> mgr;
    private int ordLimit;
    private AtomicBoolean stop;

    public IndexerThread(
        IndexWriter w,
        FacetsConfig config,
        TaxonomyWriter tw,
        ReferenceManager<SearcherAndTaxonomy> mgr,
        int ordLimit,
        AtomicBoolean stop) {
      this.w = w;
      this.config = config;
      this.tw = tw;
      this.mgr = mgr;
      this.ordLimit = ordLimit;
      this.stop = stop;
    }

    @Override
    public void run() {
      try {
        Set<String> seen = new HashSet<>();
        List<String> paths = new ArrayList<>();
        while (true) {
          Document doc = new Document();
          int numPaths = TestUtil.nextInt(random(), 1, 5);
          for (int i = 0; i < numPaths; i++) {
            String path;
            if (!paths.isEmpty() && random().nextInt(5) != 4) {
              // Use previous path
              path = paths.get(random().nextInt(paths.size()));
            } else {
              // Create new path
              path = null;
              while (true) {
                path = TestUtil.randomRealisticUnicodeString(random());
                if (path.length() != 0 && !seen.contains(path)) {
                  seen.add(path);
                  paths.add(path);
                  break;
                }
              }
            }
            doc.add(new FacetField("field", path));
          }
          try {
            w.addDocument(config.build(tw, doc));
            if (mgr != null && random().nextDouble() < 0.02) {
              w.commit();
              tw.commit();
              mgr.maybeRefresh();
            }
          } catch (IOException ioe) {
            throw new RuntimeException(ioe);
          }

          if (VERBOSE) {
            System.out.println("TW size=" + tw.getSize() + " vs " + ordLimit);
          }

          if (tw.getSize() >= ordLimit) {
            break;
          }
        }
      } finally {
        stop.set(true);
      }
    }
  }

  @SuppressForbidden(reason = "Thread sleep")
  public void testNRT() throws Exception {
    Directory dir = newDirectory();
    Directory taxoDir = newDirectory();
    IndexWriterConfig iwc = newIndexWriterConfig(new MockAnalyzer(random()));
    // Don't allow tiny maxBufferedDocs; it can make this
    // test too slow:
    iwc.setMaxBufferedDocs(Math.max(500, iwc.getMaxBufferedDocs()));

    // MockRandom/AlcololicMergePolicy are too slow:
    TieredMergePolicy tmp = new TieredMergePolicy();
    tmp.setFloorSegmentMB(.001);
    iwc.setMergePolicy(tmp);
    final IndexWriter w = new IndexWriter(dir, iwc);
    final DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(taxoDir);
    final FacetsConfig config = new FacetsConfig();
    config.setMultiValued("field", true);
    final AtomicBoolean stop = new AtomicBoolean();

    // How many unique facets to index before stopping:
    final int ordLimit = TEST_NIGHTLY ? 100000 : 6000;

    Thread indexer = new IndexerThread(w, config, tw, null, ordLimit, stop);

    final SearcherTaxonomyManager mgr = new SearcherTaxonomyManager(w, true, null, tw);

    Thread reopener =
        new Thread() {
          @SuppressForbidden(reason = "Thread sleep")
          @Override
          public void run() {
            while (!stop.get()) {
              try {
                // Sleep for up to 20 msec:
                Thread.sleep(random().nextInt(20));

                if (VERBOSE) {
                  System.out.println("TEST: reopen");
                }

                mgr.maybeRefresh();

                if (VERBOSE) {
                  System.out.println("TEST: reopen done");
                }
              } catch (Exception ioe) {
                throw new RuntimeException(ioe);
              }
            }
          }
        };

    reopener.setName("reopener");
    reopener.start();

    indexer.setName("indexer");
    indexer.start();

    try {
      while (!stop.get()) {
        SearcherAndTaxonomy pair = mgr.acquire();
        try {
          // System.out.println("search maxOrd=" + pair.taxonomyReader.getSize());
          FacetsCollector sfc =
              pair.searcher().search(new MatchAllDocsQuery(), new FacetsCollectorManager());
          Facets facets = getTaxonomyFacetCounts(pair.taxonomyReader(), config, sfc);
          FacetResult result = facets.getTopChildren(10, "field");
          if (pair.searcher().getIndexReader().numDocs() > 0) {
            // System.out.println(pair.taxonomyReader.getSize());
            assertTrue(result.childCount > 0);
            assertTrue(result.labelValues.length > 0);
          }

          // if (VERBOSE) {
          // System.out.println("TEST: facets=" + FacetTestUtils.toString(results.get(0)));
          // }
        } finally {
          mgr.release(pair);
        }
      }
    } finally {
      indexer.join();
      reopener.join();
    }

    if (VERBOSE) {
      System.out.println("TEST: now stop");
    }

    w.close();
    IOUtils.close(mgr, tw, taxoDir, dir);
  }

  public void testDirectory() throws Exception {
    Directory indexDir = newDirectory();
    Directory taxoDir = newDirectory();
    final IndexWriter w =
        new IndexWriter(indexDir, newIndexWriterConfig(new MockAnalyzer(random())));
    final DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(taxoDir);
    // first empty commit
    w.commit();
    tw.commit();
    final SearcherTaxonomyManager mgr = new SearcherTaxonomyManager(indexDir, taxoDir, null);
    final FacetsConfig config = new FacetsConfig();
    config.setMultiValued("field", true);
    final AtomicBoolean stop = new AtomicBoolean();

    // How many unique facets to index before stopping:
    final int ordLimit = TEST_NIGHTLY ? 100000 : 600;

    Thread indexer = new IndexerThread(w, config, tw, mgr, ordLimit, stop);
    indexer.start();

    try {
      while (!stop.get()) {
        SearcherAndTaxonomy pair = mgr.acquire();
        try {
          // System.out.println("search maxOrd=" + pair.taxonomyReader.getSize());
          FacetsCollector sfc =
              pair.searcher().search(new MatchAllDocsQuery(), new FacetsCollectorManager());
          Facets facets = getTaxonomyFacetCounts(pair.taxonomyReader(), config, sfc);
          FacetResult result = facets.getTopChildren(10, "field");
          if (pair.searcher().getIndexReader().numDocs() > 0) {
            // System.out.println(pair.taxonomyReader.getSize());
            assertTrue(result.childCount > 0);
            assertTrue(result.labelValues.length > 0);
          }

          // if (VERBOSE) {
          // System.out.println("TEST: facets=" + FacetTestUtils.toString(results.get(0)));
          // }
        } finally {
          mgr.release(pair);
        }
      }
    } finally {
      indexer.join();
    }

    if (VERBOSE) {
      System.out.println("TEST: now stop");
    }

    w.close();
    IOUtils.close(mgr, tw, taxoDir, indexDir);
  }

  public void testReplaceTaxonomyNRT() throws Exception {
    Directory dir = newDirectory();
    Directory taxoDir = newDirectory();
    IndexWriter w = new IndexWriter(dir, newIndexWriterConfig(new MockAnalyzer(random())));
    DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(taxoDir);

    Directory taxoDir2 = newDirectory();
    DirectoryTaxonomyWriter tw2 = new DirectoryTaxonomyWriter(taxoDir2);
    tw2.close();

    SearcherTaxonomyManager mgr = new SearcherTaxonomyManager(w, true, null, tw);
    w.addDocument(new Document());
    tw.replaceTaxonomy(taxoDir2);
    taxoDir2.close();

    expectThrows(IllegalStateException.class, mgr::maybeRefresh);

    w.close();
    IOUtils.close(mgr, tw, taxoDir, dir);
  }

  public void testReplaceTaxonomyDirectory() throws Exception {
    Directory indexDir = newDirectory();
    Directory taxoDir = newDirectory();
    IndexWriter w = new IndexWriter(indexDir, newIndexWriterConfig(new MockAnalyzer(random())));
    DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(taxoDir);
    w.commit();
    tw.commit();

    Directory taxoDir2 = newDirectory();
    DirectoryTaxonomyWriter tw2 = new DirectoryTaxonomyWriter(taxoDir2);
    tw2.addCategory(new FacetLabel("a", "b"));
    tw2.close();

    SearcherTaxonomyManager mgr = new SearcherTaxonomyManager(indexDir, taxoDir, null);
    SearcherAndTaxonomy pair = mgr.acquire();
    try {
      assertEquals(1, pair.taxonomyReader().getSize());
    } finally {
      mgr.release(pair);
    }

    w.addDocument(new Document());
    tw.replaceTaxonomy(taxoDir2);
    taxoDir2.close();
    w.commit();
    tw.commit();

    mgr.maybeRefresh();
    pair = mgr.acquire();
    try {
      assertEquals(3, pair.taxonomyReader().getSize());
    } finally {
      mgr.release(pair);
    }

    w.close();
    IOUtils.close(mgr, tw, taxoDir, indexDir);
  }

  public void testExceptionDuringRefresh() throws Exception {

    Directory indexDir = newDirectory();
    Directory taxoDir = newDirectory();

    IndexWriter w = new IndexWriter(indexDir, newIndexWriterConfig(new MockAnalyzer(random())));
    DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(taxoDir);
    w.commit();
    tw.commit();

    SearcherTaxonomyManager mgr = new SearcherTaxonomyManager(indexDir, taxoDir, null);

    tw.addCategory(new FacetLabel("a", "b"));
    w.addDocument(new Document());

    tw.commit();
    w.commit();

    // intentionally corrupt the taxo index:
    SegmentInfos infos = SegmentInfos.readLatestCommit(taxoDir);
    taxoDir.deleteFile(infos.getSegmentsFileName());
    expectThrows(IndexNotFoundException.class, mgr::maybeRefreshBlocking);
    IOUtils.close(w, tw, mgr, indexDir, taxoDir);
  }

  /** Returns the first commit with generation higher than current reader commit */
  public static class NextCommitSelector implements RefreshCommitSupplier {
    @Override
    public IndexCommit getSearcherRefreshCommit(DirectoryReader reader) throws IOException {
      List<IndexCommit> commits = DirectoryReader.listCommits(reader.directory());
      IndexCommit current = reader.getIndexCommit();
      for (int i = 0; i < commits.size(); i++) {
        IndexCommit commit = commits.get(i);
        if (commit.getGeneration() > current.getGeneration()) {
          return commit;
        }
      }
      // we're already on latest commit
      return null;
    }
  }

  public void testStepWiseCommitRefresh() throws Exception {
    Directory dir = newDirectory();
    Directory taxoDir = newDirectory();
    IndexWriter w =
        new IndexWriter(
            dir,
            newIndexWriterConfig(new MockAnalyzer(random()))
                .setIndexDeletionPolicy(NoDeletionPolicy.INSTANCE));
    DirectoryTaxonomyWriter tw = new DirectoryTaxonomyWriter(taxoDir);
    int docId = 0;
    // create initial commit
    for (int i = 0; i < 20; i++) {
      Document doc = new Document();
      doc.add(newStringField("docId", "doc-" + docId++, Field.Store.YES));
      w.addDocument(doc);
    }
    w.commit();
    tw.commit();

    int colorIndex = 0;
    final String[] colors = new String[] {"red", "green", "blue", "yellow"};
    FacetsConfig config = new FacetsConfig();
    DirectoryTaxonomyReader taxoReader = new DirectoryTaxonomyReader(taxoDir);
    SearcherTaxonomyManager sat =
        new SearcherTaxonomyManager(
            DirectoryReader.open(dir), taxoReader, null, new NextCommitSelector());

    final int numCommits = 5;
    for (int i = 0; i < numCommits; i++) {
      for (int j = 0; j < 20; j++) {
        Document doc = new Document();
        doc.add(newStringField("docId", "doc-" + docId++, Field.Store.YES));
        doc.add(new FacetField("Color", colors[colorIndex++ % colors.length]));
        doc = config.build(tw, doc);
        w.addDocument(doc);
      }
      w.commit();
      tw.commit();
    }

    // maybeRefresh only refreshes on the next incremental commit
    // so it takes us numCommits to get to latest
    int stepsToCurrent = 0;
    while (sat.isSearcherCurrent() == false) {
      long oldGen = sat.getSearcherCommitGeneration();
      sat.maybeRefreshBlocking();
      long newGen = sat.getSearcherCommitGeneration();
      assertEquals(newGen, oldGen + 1);
      stepsToCurrent++;
      assertTrue(sat.isTaxonomyCurrent());
    }
    assertEquals(numCommits, stepsToCurrent);
    sat.close();
    w.close();
    tw.close();
    dir.close();
    taxoDir.close();
  }
}
