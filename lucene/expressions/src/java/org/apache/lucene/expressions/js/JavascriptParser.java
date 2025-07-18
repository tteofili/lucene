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

// ANTLR GENERATED CODE: DO NOT EDIT.

package org.apache.lucene.expressions.js;

import java.util.List;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;

@SuppressWarnings({
  "all",
  "warnings",
  "unchecked",
  "unused",
  "cast",
  "CheckReturnValue",
  "this-escape"
})
class JavascriptParser extends Parser {
  static {
    RuntimeMetaData.checkVersion("4.13.2", RuntimeMetaData.VERSION);
  }

  protected static final DFA[] _decisionToDFA;
  protected static final PredictionContextCache _sharedContextCache = new PredictionContextCache();
  public static final int LP = 1,
      RP = 2,
      COMMA = 3,
      BOOLNOT = 4,
      BWNOT = 5,
      MUL = 6,
      DIV = 7,
      REM = 8,
      ADD = 9,
      SUB = 10,
      LSH = 11,
      RSH = 12,
      USH = 13,
      LT = 14,
      LTE = 15,
      GT = 16,
      GTE = 17,
      EQ = 18,
      NE = 19,
      BWAND = 20,
      BWXOR = 21,
      BWOR = 22,
      BOOLAND = 23,
      BOOLOR = 24,
      COND = 25,
      COLON = 26,
      WS = 27,
      VARIABLE = 28,
      OCTAL = 29,
      HEX = 30,
      DECIMAL = 31;
  public static final int RULE_compile = 0, RULE_expression = 1;

  private static String[] makeRuleNames() {
    return new String[] {"compile", "expression"};
  }

  public static final String[] ruleNames = makeRuleNames();

  private static String[] makeLiteralNames() {
    return new String[] {
      null, null, null, null, null, null, null, null, null, null, null, "'<<'", "'>>'", "'>>>'",
      null, "'<='", null, "'>='", "'=='", "'!='", null, null, null, "'&&'", "'||'"
    };
  }

  private static final String[] _LITERAL_NAMES = makeLiteralNames();

  private static String[] makeSymbolicNames() {
    return new String[] {
      null,
      "LP",
      "RP",
      "COMMA",
      "BOOLNOT",
      "BWNOT",
      "MUL",
      "DIV",
      "REM",
      "ADD",
      "SUB",
      "LSH",
      "RSH",
      "USH",
      "LT",
      "LTE",
      "GT",
      "GTE",
      "EQ",
      "NE",
      "BWAND",
      "BWXOR",
      "BWOR",
      "BOOLAND",
      "BOOLOR",
      "COND",
      "COLON",
      "WS",
      "VARIABLE",
      "OCTAL",
      "HEX",
      "DECIMAL"
    };
  }

  private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
  public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

  /**
   * @deprecated Use {@link #VOCABULARY} instead.
   */
  @Deprecated public static final String[] tokenNames;

  static {
    tokenNames = new String[_SYMBOLIC_NAMES.length];
    for (int i = 0; i < tokenNames.length; i++) {
      tokenNames[i] = VOCABULARY.getLiteralName(i);
      if (tokenNames[i] == null) {
        tokenNames[i] = VOCABULARY.getSymbolicName(i);
      }

      if (tokenNames[i] == null) {
        tokenNames[i] = "<INVALID>";
      }
    }
  }

  @Override
  @Deprecated
  public String[] getTokenNames() {
    return tokenNames;
  }

  @Override
  public Vocabulary getVocabulary() {
    return VOCABULARY;
  }

  @Override
  public String getGrammarFileName() {
    return "Javascript.g4";
  }

  @Override
  public String[] getRuleNames() {
    return ruleNames;
  }

  @Override
  public String getSerializedATN() {
    return _serializedATN;
  }

  @Override
  public ATN getATN() {
    return _ATN;
  }

  public JavascriptParser(TokenStream input) {
    super(input);
    _interp = new ParserATNSimulator(this, _ATN, _decisionToDFA, _sharedContextCache);
  }

  @SuppressWarnings("CheckReturnValue")
  public static class CompileContext extends ParserRuleContext {
    public ExpressionContext expression() {
      return getRuleContext(ExpressionContext.class, 0);
    }

    public TerminalNode EOF() {
      return getToken(JavascriptParser.EOF, 0);
    }

    public CompileContext(ParserRuleContext parent, int invokingState) {
      super(parent, invokingState);
    }

    @Override
    public int getRuleIndex() {
      return RULE_compile;
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitCompile(this);
      else return visitor.visitChildren(this);
    }
  }

  public final CompileContext compile() throws RecognitionException {
    CompileContext _localctx = new CompileContext(_ctx, getState());
    enterRule(_localctx, 0, RULE_compile);
    try {
      enterOuterAlt(_localctx, 1);
      {
        setState(4);
        expression(0);
        setState(5);
        match(EOF);
      }
    } catch (RecognitionException re) {
      _localctx.exception = re;
      _errHandler.reportError(this, re);
      _errHandler.recover(this, re);
    } finally {
      exitRule();
    }
    return _localctx;
  }

  @SuppressWarnings("CheckReturnValue")
  public static class ExpressionContext extends ParserRuleContext {
    public ExpressionContext(ParserRuleContext parent, int invokingState) {
      super(parent, invokingState);
    }

    @Override
    public int getRuleIndex() {
      return RULE_expression;
    }

    public ExpressionContext() {}

    public void copyFrom(ExpressionContext ctx) {
      super.copyFrom(ctx);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class ConditionalContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode COND() {
      return getToken(JavascriptParser.COND, 0);
    }

    public TerminalNode COLON() {
      return getToken(JavascriptParser.COLON, 0);
    }

    public ConditionalContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitConditional(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BoolorContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode BOOLOR() {
      return getToken(JavascriptParser.BOOLOR, 0);
    }

    public BoolorContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBoolor(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BoolcompContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode LT() {
      return getToken(JavascriptParser.LT, 0);
    }

    public TerminalNode LTE() {
      return getToken(JavascriptParser.LTE, 0);
    }

    public TerminalNode GT() {
      return getToken(JavascriptParser.GT, 0);
    }

    public TerminalNode GTE() {
      return getToken(JavascriptParser.GTE, 0);
    }

    public BoolcompContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBoolcomp(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class NumericContext extends ExpressionContext {
    public TerminalNode OCTAL() {
      return getToken(JavascriptParser.OCTAL, 0);
    }

    public TerminalNode HEX() {
      return getToken(JavascriptParser.HEX, 0);
    }

    public TerminalNode DECIMAL() {
      return getToken(JavascriptParser.DECIMAL, 0);
    }

    public NumericContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitNumeric(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class AddsubContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode ADD() {
      return getToken(JavascriptParser.ADD, 0);
    }

    public TerminalNode SUB() {
      return getToken(JavascriptParser.SUB, 0);
    }

    public AddsubContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitAddsub(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class UnaryContext extends ExpressionContext {
    public ExpressionContext expression() {
      return getRuleContext(ExpressionContext.class, 0);
    }

    public TerminalNode BOOLNOT() {
      return getToken(JavascriptParser.BOOLNOT, 0);
    }

    public TerminalNode BWNOT() {
      return getToken(JavascriptParser.BWNOT, 0);
    }

    public TerminalNode ADD() {
      return getToken(JavascriptParser.ADD, 0);
    }

    public TerminalNode SUB() {
      return getToken(JavascriptParser.SUB, 0);
    }

    public UnaryContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitUnary(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class PrecedenceContext extends ExpressionContext {
    public TerminalNode LP() {
      return getToken(JavascriptParser.LP, 0);
    }

    public ExpressionContext expression() {
      return getRuleContext(ExpressionContext.class, 0);
    }

    public TerminalNode RP() {
      return getToken(JavascriptParser.RP, 0);
    }

    public PrecedenceContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitPrecedence(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class MuldivContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode MUL() {
      return getToken(JavascriptParser.MUL, 0);
    }

    public TerminalNode DIV() {
      return getToken(JavascriptParser.DIV, 0);
    }

    public TerminalNode REM() {
      return getToken(JavascriptParser.REM, 0);
    }

    public MuldivContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitMuldiv(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class ExternalContext extends ExpressionContext {
    public TerminalNode VARIABLE() {
      return getToken(JavascriptParser.VARIABLE, 0);
    }

    public TerminalNode LP() {
      return getToken(JavascriptParser.LP, 0);
    }

    public TerminalNode RP() {
      return getToken(JavascriptParser.RP, 0);
    }

    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public List<TerminalNode> COMMA() {
      return getTokens(JavascriptParser.COMMA);
    }

    public TerminalNode COMMA(int i) {
      return getToken(JavascriptParser.COMMA, i);
    }

    public ExternalContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitExternal(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BwshiftContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode LSH() {
      return getToken(JavascriptParser.LSH, 0);
    }

    public TerminalNode RSH() {
      return getToken(JavascriptParser.RSH, 0);
    }

    public TerminalNode USH() {
      return getToken(JavascriptParser.USH, 0);
    }

    public BwshiftContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBwshift(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BworContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode BWOR() {
      return getToken(JavascriptParser.BWOR, 0);
    }

    public BworContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBwor(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BoolandContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode BOOLAND() {
      return getToken(JavascriptParser.BOOLAND, 0);
    }

    public BoolandContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBooland(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BwxorContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode BWXOR() {
      return getToken(JavascriptParser.BWXOR, 0);
    }

    public BwxorContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBwxor(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BwandContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode BWAND() {
      return getToken(JavascriptParser.BWAND, 0);
    }

    public BwandContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBwand(this);
      else return visitor.visitChildren(this);
    }
  }

  @SuppressWarnings("CheckReturnValue")
  public static class BooleqneContext extends ExpressionContext {
    public List<ExpressionContext> expression() {
      return getRuleContexts(ExpressionContext.class);
    }

    public ExpressionContext expression(int i) {
      return getRuleContext(ExpressionContext.class, i);
    }

    public TerminalNode EQ() {
      return getToken(JavascriptParser.EQ, 0);
    }

    public TerminalNode NE() {
      return getToken(JavascriptParser.NE, 0);
    }

    public BooleqneContext(ExpressionContext ctx) {
      copyFrom(ctx);
    }

    @Override
    public <T> T accept(ParseTreeVisitor<? extends T> visitor) {
      if (visitor instanceof JavascriptVisitor)
        return ((JavascriptVisitor<? extends T>) visitor).visitBooleqne(this);
      else return visitor.visitChildren(this);
    }
  }

  public final ExpressionContext expression() throws RecognitionException {
    return expression(0);
  }

  private ExpressionContext expression(int _p) throws RecognitionException {
    ParserRuleContext _parentctx = _ctx;
    int _parentState = getState();
    ExpressionContext _localctx = new ExpressionContext(_ctx, _parentState);
    ExpressionContext _prevctx = _localctx;
    int _startState = 2;
    enterRecursionRule(_localctx, 2, RULE_expression, _p);
    int _la;
    try {
      int _alt;
      enterOuterAlt(_localctx, 1);
      {
        setState(30);
        _errHandler.sync(this);
        switch (_input.LA(1)) {
          case LP:
            {
              _localctx = new PrecedenceContext(_localctx);
              _ctx = _localctx;
              _prevctx = _localctx;

              setState(8);
              match(LP);
              setState(9);
              expression(0);
              setState(10);
              match(RP);
            }
            break;
          case OCTAL:
          case HEX:
          case DECIMAL:
            {
              _localctx = new NumericContext(_localctx);
              _ctx = _localctx;
              _prevctx = _localctx;
              setState(12);
              _la = _input.LA(1);
              if (!((((_la) & ~0x3f) == 0 && ((1L << _la) & 3758096384L) != 0))) {
                _errHandler.recoverInline(this);
              } else {
                if (_input.LA(1) == Token.EOF) matchedEOF = true;
                _errHandler.reportMatch(this);
                consume();
              }
            }
            break;
          case VARIABLE:
            {
              _localctx = new ExternalContext(_localctx);
              _ctx = _localctx;
              _prevctx = _localctx;
              setState(13);
              match(VARIABLE);
              setState(26);
              _errHandler.sync(this);
              switch (getInterpreter().adaptivePredict(_input, 2, _ctx)) {
                case 1:
                  {
                    setState(14);
                    match(LP);
                    setState(23);
                    _errHandler.sync(this);
                    _la = _input.LA(1);
                    if ((((_la) & ~0x3f) == 0 && ((1L << _la) & 4026533426L) != 0)) {
                      {
                        setState(15);
                        expression(0);
                        setState(20);
                        _errHandler.sync(this);
                        _la = _input.LA(1);
                        while (_la == COMMA) {
                          {
                            {
                              setState(16);
                              match(COMMA);
                              setState(17);
                              expression(0);
                            }
                          }
                          setState(22);
                          _errHandler.sync(this);
                          _la = _input.LA(1);
                        }
                      }
                    }

                    setState(25);
                    match(RP);
                  }
                  break;
              }
            }
            break;
          case BOOLNOT:
          case BWNOT:
          case ADD:
          case SUB:
            {
              _localctx = new UnaryContext(_localctx);
              _ctx = _localctx;
              _prevctx = _localctx;
              setState(28);
              _la = _input.LA(1);
              if (!((((_la) & ~0x3f) == 0 && ((1L << _la) & 1584L) != 0))) {
                _errHandler.recoverInline(this);
              } else {
                if (_input.LA(1) == Token.EOF) matchedEOF = true;
                _errHandler.reportMatch(this);
                consume();
              }
              setState(29);
              expression(12);
            }
            break;
          default:
            throw new NoViableAltException(this);
        }
        _ctx.stop = _input.LT(-1);
        setState(70);
        _errHandler.sync(this);
        _alt = getInterpreter().adaptivePredict(_input, 5, _ctx);
        while (_alt != 2 && _alt != org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER) {
          if (_alt == 1) {
            if (_parseListeners != null) triggerExitRuleEvent();
            _prevctx = _localctx;
            {
              setState(68);
              _errHandler.sync(this);
              switch (getInterpreter().adaptivePredict(_input, 4, _ctx)) {
                case 1:
                  {
                    _localctx = new MuldivContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(32);
                    if (!(precpred(_ctx, 11)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 11)");
                    setState(33);
                    _la = _input.LA(1);
                    if (!((((_la) & ~0x3f) == 0 && ((1L << _la) & 448L) != 0))) {
                      _errHandler.recoverInline(this);
                    } else {
                      if (_input.LA(1) == Token.EOF) matchedEOF = true;
                      _errHandler.reportMatch(this);
                      consume();
                    }
                    setState(34);
                    expression(12);
                  }
                  break;
                case 2:
                  {
                    _localctx = new AddsubContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(35);
                    if (!(precpred(_ctx, 10)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 10)");
                    setState(36);
                    _la = _input.LA(1);
                    if (!(_la == ADD || _la == SUB)) {
                      _errHandler.recoverInline(this);
                    } else {
                      if (_input.LA(1) == Token.EOF) matchedEOF = true;
                      _errHandler.reportMatch(this);
                      consume();
                    }
                    setState(37);
                    expression(11);
                  }
                  break;
                case 3:
                  {
                    _localctx = new BwshiftContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(38);
                    if (!(precpred(_ctx, 9)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 9)");
                    setState(39);
                    _la = _input.LA(1);
                    if (!((((_la) & ~0x3f) == 0 && ((1L << _la) & 14336L) != 0))) {
                      _errHandler.recoverInline(this);
                    } else {
                      if (_input.LA(1) == Token.EOF) matchedEOF = true;
                      _errHandler.reportMatch(this);
                      consume();
                    }
                    setState(40);
                    expression(10);
                  }
                  break;
                case 4:
                  {
                    _localctx =
                        new BoolcompContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(41);
                    if (!(precpred(_ctx, 8)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 8)");
                    setState(42);
                    _la = _input.LA(1);
                    if (!((((_la) & ~0x3f) == 0 && ((1L << _la) & 245760L) != 0))) {
                      _errHandler.recoverInline(this);
                    } else {
                      if (_input.LA(1) == Token.EOF) matchedEOF = true;
                      _errHandler.reportMatch(this);
                      consume();
                    }
                    setState(43);
                    expression(9);
                  }
                  break;
                case 5:
                  {
                    _localctx =
                        new BooleqneContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(44);
                    if (!(precpred(_ctx, 7)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 7)");
                    setState(45);
                    _la = _input.LA(1);
                    if (!(_la == EQ || _la == NE)) {
                      _errHandler.recoverInline(this);
                    } else {
                      if (_input.LA(1) == Token.EOF) matchedEOF = true;
                      _errHandler.reportMatch(this);
                      consume();
                    }
                    setState(46);
                    expression(8);
                  }
                  break;
                case 6:
                  {
                    _localctx = new BwandContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(47);
                    if (!(precpred(_ctx, 6)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 6)");
                    setState(48);
                    match(BWAND);
                    setState(49);
                    expression(7);
                  }
                  break;
                case 7:
                  {
                    _localctx = new BwxorContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(50);
                    if (!(precpred(_ctx, 5)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 5)");
                    setState(51);
                    match(BWXOR);
                    setState(52);
                    expression(6);
                  }
                  break;
                case 8:
                  {
                    _localctx = new BworContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(53);
                    if (!(precpred(_ctx, 4)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 4)");
                    setState(54);
                    match(BWOR);
                    setState(55);
                    expression(5);
                  }
                  break;
                case 9:
                  {
                    _localctx = new BoolandContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(56);
                    if (!(precpred(_ctx, 3)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 3)");
                    setState(57);
                    match(BOOLAND);
                    setState(58);
                    expression(4);
                  }
                  break;
                case 10:
                  {
                    _localctx = new BoolorContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(59);
                    if (!(precpred(_ctx, 2)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 2)");
                    setState(60);
                    match(BOOLOR);
                    setState(61);
                    expression(3);
                  }
                  break;
                case 11:
                  {
                    _localctx =
                        new ConditionalContext(new ExpressionContext(_parentctx, _parentState));
                    pushNewRecursionContext(_localctx, _startState, RULE_expression);
                    setState(62);
                    if (!(precpred(_ctx, 1)))
                      throw new FailedPredicateException(this, "precpred(_ctx, 1)");
                    setState(63);
                    match(COND);
                    setState(64);
                    expression(0);
                    setState(65);
                    match(COLON);
                    setState(66);
                    expression(1);
                  }
                  break;
              }
            }
          }
          setState(72);
          _errHandler.sync(this);
          _alt = getInterpreter().adaptivePredict(_input, 5, _ctx);
        }
      }
    } catch (RecognitionException re) {
      _localctx.exception = re;
      _errHandler.reportError(this, re);
      _errHandler.recover(this, re);
    } finally {
      unrollRecursionContexts(_parentctx);
    }
    return _localctx;
  }

  public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
    switch (ruleIndex) {
      case 1:
        return expression_sempred((ExpressionContext) _localctx, predIndex);
    }
    return true;
  }

  private boolean expression_sempred(ExpressionContext _localctx, int predIndex) {
    switch (predIndex) {
      case 0:
        return precpred(_ctx, 11);
      case 1:
        return precpred(_ctx, 10);
      case 2:
        return precpred(_ctx, 9);
      case 3:
        return precpred(_ctx, 8);
      case 4:
        return precpred(_ctx, 7);
      case 5:
        return precpred(_ctx, 6);
      case 6:
        return precpred(_ctx, 5);
      case 7:
        return precpred(_ctx, 4);
      case 8:
        return precpred(_ctx, 3);
      case 9:
        return precpred(_ctx, 2);
      case 10:
        return precpred(_ctx, 1);
    }
    return true;
  }

  public static final String _serializedATN =
      "\u0004\u0001\u001fJ\u0002\u0000\u0007\u0000\u0002\u0001\u0007\u0001\u0001"
          + "\u0000\u0001\u0000\u0001\u0000\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0005\u0001\u0013\b\u0001\n\u0001\f\u0001\u0016\t\u0001"
          + "\u0003\u0001\u0018\b\u0001\u0001\u0001\u0003\u0001\u001b\b\u0001\u0001"
          + "\u0001\u0001\u0001\u0003\u0001\u001f\b\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0001"
          + "\u0001\u0001\u0001\u0001\u0001\u0001\u0001\u0005\u0001E\b\u0001\n\u0001"
          + "\f\u0001H\t\u0001\u0001\u0001\u0000\u0001\u0002\u0002\u0000\u0002\u0000"
          + "\u0007\u0001\u0000\u001d\u001f\u0002\u0000\u0004\u0005\t\n\u0001\u0000"
          + "\u0006\b\u0001\u0000\t\n\u0001\u0000\u000b\r\u0001\u0000\u000e\u0011\u0001"
          + "\u0000\u0012\u0013X\u0000\u0004\u0001\u0000\u0000\u0000\u0002\u001e\u0001"
          + "\u0000\u0000\u0000\u0004\u0005\u0003\u0002\u0001\u0000\u0005\u0006\u0005"
          + "\u0000\u0000\u0001\u0006\u0001\u0001\u0000\u0000\u0000\u0007\b\u0006\u0001"
          + "\uffff\uffff\u0000\b\t\u0005\u0001\u0000\u0000\t\n\u0003\u0002\u0001\u0000"
          + "\n\u000b\u0005\u0002\u0000\u0000\u000b\u001f\u0001\u0000\u0000\u0000\f"
          + "\u001f\u0007\u0000\u0000\u0000\r\u001a\u0005\u001c\u0000\u0000\u000e\u0017"
          + "\u0005\u0001\u0000\u0000\u000f\u0014\u0003\u0002\u0001\u0000\u0010\u0011"
          + "\u0005\u0003\u0000\u0000\u0011\u0013\u0003\u0002\u0001\u0000\u0012\u0010"
          + "\u0001\u0000\u0000\u0000\u0013\u0016\u0001\u0000\u0000\u0000\u0014\u0012"
          + "\u0001\u0000\u0000\u0000\u0014\u0015\u0001\u0000\u0000\u0000\u0015\u0018"
          + "\u0001\u0000\u0000\u0000\u0016\u0014\u0001\u0000\u0000\u0000\u0017\u000f"
          + "\u0001\u0000\u0000\u0000\u0017\u0018\u0001\u0000\u0000\u0000\u0018\u0019"
          + "\u0001\u0000\u0000\u0000\u0019\u001b\u0005\u0002\u0000\u0000\u001a\u000e"
          + "\u0001\u0000\u0000\u0000\u001a\u001b\u0001\u0000\u0000\u0000\u001b\u001f"
          + "\u0001\u0000\u0000\u0000\u001c\u001d\u0007\u0001\u0000\u0000\u001d\u001f"
          + "\u0003\u0002\u0001\f\u001e\u0007\u0001\u0000\u0000\u0000\u001e\f\u0001"
          + "\u0000\u0000\u0000\u001e\r\u0001\u0000\u0000\u0000\u001e\u001c\u0001\u0000"
          + "\u0000\u0000\u001fF\u0001\u0000\u0000\u0000 !\n\u000b\u0000\u0000!\"\u0007"
          + "\u0002\u0000\u0000\"E\u0003\u0002\u0001\f#$\n\n\u0000\u0000$%\u0007\u0003"
          + "\u0000\u0000%E\u0003\u0002\u0001\u000b&\'\n\t\u0000\u0000\'(\u0007\u0004"
          + "\u0000\u0000(E\u0003\u0002\u0001\n)*\n\b\u0000\u0000*+\u0007\u0005\u0000"
          + "\u0000+E\u0003\u0002\u0001\t,-\n\u0007\u0000\u0000-.\u0007\u0006\u0000"
          + "\u0000.E\u0003\u0002\u0001\b/0\n\u0006\u0000\u000001\u0005\u0014\u0000"
          + "\u00001E\u0003\u0002\u0001\u000723\n\u0005\u0000\u000034\u0005\u0015\u0000"
          + "\u00004E\u0003\u0002\u0001\u000656\n\u0004\u0000\u000067\u0005\u0016\u0000"
          + "\u00007E\u0003\u0002\u0001\u000589\n\u0003\u0000\u00009:\u0005\u0017\u0000"
          + "\u0000:E\u0003\u0002\u0001\u0004;<\n\u0002\u0000\u0000<=\u0005\u0018\u0000"
          + "\u0000=E\u0003\u0002\u0001\u0003>?\n\u0001\u0000\u0000?@\u0005\u0019\u0000"
          + "\u0000@A\u0003\u0002\u0001\u0000AB\u0005\u001a\u0000\u0000BC\u0003\u0002"
          + "\u0001\u0001CE\u0001\u0000\u0000\u0000D \u0001\u0000\u0000\u0000D#\u0001"
          + "\u0000\u0000\u0000D&\u0001\u0000\u0000\u0000D)\u0001\u0000\u0000\u0000"
          + "D,\u0001\u0000\u0000\u0000D/\u0001\u0000\u0000\u0000D2\u0001\u0000\u0000"
          + "\u0000D5\u0001\u0000\u0000\u0000D8\u0001\u0000\u0000\u0000D;\u0001\u0000"
          + "\u0000\u0000D>\u0001\u0000\u0000\u0000EH\u0001\u0000\u0000\u0000FD\u0001"
          + "\u0000\u0000\u0000FG\u0001\u0000\u0000\u0000G\u0003\u0001\u0000\u0000"
          + "\u0000HF\u0001\u0000\u0000\u0000\u0006\u0014\u0017\u001a\u001eDF";
  public static final ATN _ATN = new ATNDeserializer().deserialize(_serializedATN.toCharArray());

  static {
    _decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
    for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
      _decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
    }
  }
}
