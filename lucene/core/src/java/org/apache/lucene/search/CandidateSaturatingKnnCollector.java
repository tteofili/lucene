package org.apache.lucene.search;

public class CandidateSaturatingKnnCollector implements KnnCollector {
    private final KnnCollector delegate;

    public CandidateSaturatingKnnCollector(KnnCollector delegate) {
        this.delegate = delegate;
    }

    @Override
    public boolean earlyTerminated() {
        return delegate.earlyTerminated();
    }

    @Override
    public void incVisitedCount(int count) {
        delegate.incVisitedCount(count);
    }

    @Override
    public long visitedCount() {
        return delegate.visitedCount();
    }

    @Override
    public long visitLimit() {
        return delegate.visitLimit();
    }

    @Override
    public int k() {
        return delegate.k();
    }

    @Override
    public boolean collect(int docId, float similarity) {
        return delegate.collect(docId, similarity);
    }

    @Override
    public float minCompetitiveSimilarity() {
        return delegate.minCompetitiveSimilarity();
    }

    @Override
    public TopDocs topDocs() {
        return delegate.topDocs();
    }
}
