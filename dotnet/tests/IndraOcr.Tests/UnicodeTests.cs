using IndraOcr.Numerics;

namespace IndraOcr.Tests;

public class UnicodeTests
{
    [Theory]
    [InlineData('a', Unicode.Bidi.L)]
    [InlineData('A', Unicode.Bidi.L)]
    [InlineData('1', Unicode.Bidi.EN)]
    [InlineData('\u0661', Unicode.Bidi.AN)] // Arabic-Indic digit one
    [InlineData('\u0627', Unicode.Bidi.AL)] // ARABIC LETTER ALEF
    [InlineData('\u05D0', Unicode.Bidi.R)] // HEBREW LETTER ALEF
    [InlineData('\u200F', Unicode.Bidi.R)] // RIGHT-TO-LEFT MARK
    [InlineData(' ', Unicode.Bidi.Other)] // whitespace is neutral
    [InlineData('\u4E2D', Unicode.Bidi.L)] // CJK ideograph is L
    public void BidiClassesMatchPython(char c, Unicode.Bidi expected)
    {
        Assert.Equal(expected, Unicode.BidiClass(c));
    }

    [Theory]
    [InlineData('a', false)]
    [InlineData('A', false)]
    [InlineData('\u4E2D', true)] // 中
    [InlineData('\uFF21', true)] // fullwidth A
    [InlineData('\u30A2', true)] // katakana A
    [InlineData('\u00E9', false)] // é
    public void EastAsianWidthMatchesPython(char c, bool wide)
    {
        Assert.Equal(wide, Unicode.IsWide(c));
    }

    [Fact]
    public void RtlAndLtrPredicatesMatchRtlPy()
    {
        Assert.True(Unicode.IsRtl('\u0627'));
        Assert.True(Unicode.IsRtl('\u05D0'));
        Assert.False(Unicode.IsRtl('a'));
        Assert.True(Unicode.IsLtr('a'));
        Assert.True(Unicode.IsLtr('7'));
        Assert.True(Unicode.IsLtr('\u0661'));
        Assert.False(Unicode.IsLtr('\u0627'));
    }

    [Fact]
    public void CodePointsJoinSurrogatePairs()
    {
        Assert.Equal(new[] { 0x1F600 }, Unicode.CodePoints("\U0001F600").ToArray());
        Assert.Equal(new[] { (int)'a', 0x1F600, (int)'b' }, Unicode.CodePoints("a\U0001F600b").ToArray());
    }
}
