using System.Runtime.InteropServices;

namespace IndraOcr.Pdf;

/// <summary>PDFium could not open the document (the code says why).</summary>
public sealed class PdfiumException : Exception
{
    public PdfiumException(string message, ulong code) : base(message)
    {
        Code = code;
    }

    public ulong Code { get; }

    private static readonly Dictionary<ulong, string> Reasons = new()
    {
        [0] = "Success",
        [1] = "Unknown error",
        [2] = "File access error",
        [3] = "Data format error",
        [4] = "Incorrect password error",
        [5] = "Unsupported security scheme error",
        [6] = "Page not found or content error",
    };

    internal static PdfiumException FromLastError()
    {
        ulong code = PdfiumNative.FPDF_GetLastError();
        string reason = Reasons.TryGetValue(code, out var r) ? r : "Unknown error";
        return new PdfiumException($"Failed to load document (PDFium: {reason}).", code);
    }
}

/// <summary>
/// An open PDF (PDFium).  PDFium is not thread-safe, so every call is serialized
/// through one gate; S6 will move PDF work to processes or a dedicated worker
/// rather than relying on this lock for throughput.
/// </summary>
public sealed class PdfDocument : IDisposable
{
    private IntPtr _doc;

    private PdfDocument(IntPtr doc)
    {
        _doc = doc;
    }

    public static PdfDocument Open(string path, string? password = null)
    {
        lock (PdfiumNative.Sync)
        {
            PdfiumNative.EnsureInitialized();
            IntPtr doc = PdfiumNative.FPDF_LoadDocument(path, password);
            if (doc == IntPtr.Zero)
            {
                throw PdfiumException.FromLastError();
            }
            return new PdfDocument(doc);
        }
    }

    public int PageCount
    {
        get
        {
            lock (PdfiumNative.Sync)
            {
                return PdfiumNative.FPDF_GetPageCount(_doc);
            }
        }
    }

    public PdfPage LoadPage(int index)
    {
        lock (PdfiumNative.Sync)
        {
            IntPtr page = PdfiumNative.FPDF_LoadPage(_doc, index);
            if (page == IntPtr.Zero)
            {
                throw new InvalidOperationException($"cannot load page {index}");
            }
            return new PdfPage(this, page);
        }
    }

    public void Dispose()
    {
        lock (PdfiumNative.Sync)
        {
            if (_doc != IntPtr.Zero)
            {
                PdfiumNative.FPDF_CloseDocument(_doc);
                _doc = IntPtr.Zero;
            }
        }
    }
}

/// <summary>One PDF page: size, rotation, rendering and text.</summary>
public sealed class PdfPage : IDisposable
{
    private IntPtr _page;
    private readonly PdfDocument _doc;

    internal PdfPage(PdfDocument doc, IntPtr page)
    {
        _doc = doc;
        _page = page;
    }

    internal IntPtr Handle
    {
        get
        {
            lock (PdfiumNative.Sync)
            {
                return _page;
            }
        }
    }

    public float Width
    {
        get
        {
            lock (PdfiumNative.Sync)
            {
                return PdfiumNative.FPDF_GetPageWidthF(_page);
            }
        }
    }

    public float Height
    {
        get
        {
            lock (PdfiumNative.Sync)
            {
                return PdfiumNative.FPDF_GetPageHeightF(_page);
            }
        }
    }

    /// <summary>Page size in points, as displayed (rotation applied).</summary>
    public (double Width, double Height) GetSize()
    {
        lock (PdfiumNative.Sync)
        {
            return (PdfiumNative.FPDF_GetPageWidthF(_page), PdfiumNative.FPDF_GetPageHeightF(_page));
        }
    }

    /// <summary>Clockwise page rotation in degrees (0, 90, 180 or 270).</summary>
    public int GetRotation()
    {
        lock (PdfiumNative.Sync)
        {
            return PdfiumNative.FPDFPage_GetRotation(_page) switch
            {
                0 => 0,
                1 => 90,
                2 => 180,
                3 => 270,
                _ => throw new InvalidOperationException("invalid page rotation"),
            };
        }
    }

    public void SetRotation(int degrees)
    {
        int value = degrees switch
        {
            0 => 0,
            90 => 1,
            180 => 2,
            270 => 3,
            _ => throw new ArgumentOutOfRangeException(nameof(degrees), "rotation must be 0, 90, 180 or 270"),
        };
        lock (PdfiumNative.Sync)
        {
            PdfiumNative.FPDFPage_SetRotation(_page, value);
        }
    }

    /// <summary>
    /// Grayscale render at scale pixels per point: ceil(size * scale), white fill,
    /// annotations drawn (pypdfium2's defaults), row major.
    /// </summary>
    public byte[] RenderGray(double scale)
    {
        lock (PdfiumNative.Sync)
        {
            int width = (int)Math.Ceiling(PdfiumNative.FPDF_GetPageWidthF(_page) * scale);
            int height = (int)Math.Ceiling(PdfiumNative.FPDF_GetPageHeightF(_page) * scale);
            if (width < 1 || height < 1)
            {
                return Array.Empty<byte>();
            }
            var buffer = new byte[width * height];
            var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            IntPtr bitmap = IntPtr.Zero;
            try
            {
                bitmap = PdfiumNative.FPDFBitmap_CreateEx(
                    width, height, PdfiumNative.FPDFBitmap_Gray, handle.AddrOfPinnedObject(), width);
                if (bitmap == IntPtr.Zero)
                {
                    throw new InvalidOperationException("FPDFBitmap_CreateEx failed");
                }
                PdfiumNative.FPDFBitmap_FillRect(bitmap, 0, 0, width, height, 0xFFFFFFFF);
                PdfiumNative.FPDF_RenderPageBitmap(
                    bitmap, _page, 0, 0, width, height, 0,
                    PdfiumNative.FPDF_ANNOT | PdfiumNative.FPDF_GRAYSCALE);
            }
            finally
            {
                if (bitmap != IntPtr.Zero)
                {
                    PdfiumNative.FPDFBitmap_Destroy(bitmap);
                }
                handle.Free();
            }
            return buffer;
        }
    }

    /// <summary>PDF user space -> rendered device pixels (pdfium's FPDF_PageToDevice).</summary>
    internal bool PageToDevice(
        int startX, int startY, int sizeX, int sizeY, int rotate,
        double pageX, double pageY, out int deviceX, out int deviceY)
    {
        lock (PdfiumNative.Sync)
        {
            return PdfiumNative.FPDF_PageToDevice(
                _page, startX, startY, sizeX, sizeY, rotate, pageX, pageY, out deviceX, out deviceY);
        }
    }

    public PdfTextPage LoadTextPage()
    {
        lock (PdfiumNative.Sync)
        {
            IntPtr tp = PdfiumNative.FPDFText_LoadPage(_page);
            if (tp == IntPtr.Zero)
            {
                throw new InvalidOperationException("FPDFText_LoadPage failed");
            }
            return new PdfTextPage(tp);
        }
    }

    /// <summary>Page objects, recursing into form XObjects (pypdfium2's get_objects).</summary>
    public IReadOnlyList<IntPtr> EnumerateObjects(int maxDepth = 4)
    {
        lock (PdfiumNative.Sync)
        {
            var found = new List<IntPtr>();
            int count = PdfiumNative.FPDFPage_CountObjects(_page);
            for (int i = 0; i < count; i++)
            {
                Walk(PdfiumNative.FPDFPage_GetObject(_page, i), 0, maxDepth, found);
            }
            return found;
        }
    }

    private static void Walk(IntPtr obj, int depth, int maxDepth, List<IntPtr> found)
    {
        if (obj == IntPtr.Zero)
        {
            return;
        }
        found.Add(obj);
        if (depth >= maxDepth || PdfiumNative.FPDFPageObj_GetType(obj) != PdfiumNative.FPDF_PAGEOBJ_FORM)
        {
            return;
        }
        int count = PdfiumNative.FPDFFormObj_CountObjects(obj);
        for (ulong i = 0; i < (ulong)count; i++)
        {
            Walk(PdfiumNative.FPDFFormObj_GetObject(obj, i), depth + 1, maxDepth, found);
        }
    }

    public void Dispose()
    {
        lock (PdfiumNative.Sync)
        {
            if (_page != IntPtr.Zero)
            {
                PdfiumNative.FPDF_ClosePage(_page);
                _page = IntPtr.Zero;
            }
        }
    }
}

/// <summary>A page's text layer: characters, angles and boxes, in content order.</summary>
public sealed class PdfTextPage : IDisposable
{
    private IntPtr _textPage;

    internal PdfTextPage(IntPtr textPage)
    {
        _textPage = textPage;
    }

    public int CountChars
    {
        get
        {
            lock (PdfiumNative.Sync)
            {
                return PdfiumNative.FPDFText_CountChars(_textPage);
            }
        }
    }

    public uint GetUnicode(int index)
    {
        lock (PdfiumNative.Sync)
        {
            return PdfiumNative.FPDFText_GetUnicode(_textPage, index);
        }
    }

    /// <summary>The character's rotation angle, in radians (pdfium's own convention).</summary>
    public float GetCharAngle(int index)
    {
        lock (PdfiumNative.Sync)
        {
            return PdfiumNative.FPDFText_GetCharAngle(_textPage, index);
        }
    }

    /// <summary>left, right, bottom, top in PDF user space (points, y up).</summary>
    public (double Left, double Right, double Bottom, double Top)? GetCharBox(int index)
    {
        lock (PdfiumNative.Sync)
        {
            if (!PdfiumNative.FPDFText_GetCharBox(
                    _textPage, index, out double l, out double r, out double b, out double t))
            {
                return null;
            }
            return (l, r, b, t);
        }
    }

    public void Dispose()
    {
        lock (PdfiumNative.Sync)
        {
            if (_textPage != IntPtr.Zero)
            {
                PdfiumNative.FPDFText_ClosePage(_textPage);
                _textPage = IntPtr.Zero;
            }
        }
    }
}
