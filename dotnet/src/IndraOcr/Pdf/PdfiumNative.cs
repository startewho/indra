using System.Runtime.InteropServices;

namespace IndraOcr.Pdf;

/// <summary>PDFium's C API (fpdfview.h, fpdf_text.h, fpdf_edit.h).</summary>
internal static class PdfiumNative
{
    internal const string Lib = "pdfium";

    internal const int FPDF_PAGEOBJ_TEXT = 1;
    internal const int FPDF_PAGEOBJ_PATH = 2;
    internal const int FPDF_PAGEOBJ_IMAGE = 3;
    internal const int FPDF_PAGEOBJ_SHADING = 4;
    internal const int FPDF_PAGEOBJ_FORM = 5;

    internal const int FPDFBitmap_Gray = 1;
    internal const int FPDF_ANNOT = 0x01;
    internal const int FPDF_GRAYSCALE = 0x08;

    internal const int FPDF_TEXTRENDERMODE_INVISIBLE = 3;

    /// <summary>PDFium is not thread-safe: every call goes through this gate.</summary>
    internal static readonly object Sync = new();

    private static readonly object InitLock = new();
    private static bool _initialized;

    internal static void EnsureInitialized()
    {
        lock (InitLock)
        {
            if (_initialized)
            {
                return;
            }
            FPDF_InitLibrary();
            _initialized = true;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct FS_SIZEF
    {
        public float Width;
        public float Height;
    }

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDF_InitLibrary();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDF_DestroyLibrary();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ulong FPDF_GetLastError();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr FPDF_LoadDocument(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string path,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string? password);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDF_CloseDocument(IntPtr document);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDF_GetPageCount(IntPtr document);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr FPDF_LoadPage(IntPtr document, int index);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDF_ClosePage(IntPtr page);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern float FPDF_GetPageWidthF(IntPtr page);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern float FPDF_GetPageHeightF(IntPtr page);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDFPage_GetRotation(IntPtr page);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDFPage_SetRotation(IntPtr page, int rotate);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr FPDFBitmap_CreateEx(int width, int height, int format, IntPtr buffer, int stride);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDFBitmap_FillRect(IntPtr bitmap, int left, int top, int width, int height, uint color);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDFBitmap_Destroy(IntPtr bitmap);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDF_RenderPageBitmap(
        IntPtr bitmap, IntPtr page, int startX, int startY, int sizeX, int sizeY, int rotate, int flags);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr FPDFText_LoadPage(IntPtr page);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void FPDFText_ClosePage(IntPtr textPage);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDFText_CountChars(IntPtr textPage);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint FPDFText_GetUnicode(IntPtr textPage, int index);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern float FPDFText_GetCharAngle(IntPtr textPage, int index);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.Bool)]
    internal static extern bool FPDFText_GetCharBox(
        IntPtr textPage, int index, out double left, out double right, out double bottom, out double top);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.Bool)]
    internal static extern bool FPDF_PageToDevice(
        IntPtr page, int startX, int startY, int sizeX, int sizeY, int rotate,
        double pageX, double pageY, out int deviceX, out int deviceY);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDFPage_CountObjects(IntPtr page);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr FPDFPage_GetObject(IntPtr page, int index);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDFPageObj_GetType(IntPtr pageObject);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDFFormObj_CountObjects(IntPtr formObject);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr FPDFFormObj_GetObject(IntPtr formObject, ulong index);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int FPDFTextObj_GetTextRenderMode(IntPtr textObject);
}
