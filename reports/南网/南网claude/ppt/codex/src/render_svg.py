"""Render SVG with system librsvg/cairo, without introducing a package environment."""
from pathlib import Path
import ctypes as C
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
rsvg=C.CDLL('librsvg-2.so.2')
cairo=C.CDLL('libcairo.so.2')
gobject=C.CDLL('libgobject-2.0.so.0')
rsvg.rsvg_handle_new_from_data.argtypes=[C.c_char_p,C.c_size_t,C.POINTER(C.c_void_p)]
rsvg.rsvg_handle_new_from_data.restype=C.c_void_p
rsvg.rsvg_handle_render_cairo.argtypes=[C.c_void_p,C.c_void_p]
rsvg.rsvg_handle_render_cairo.restype=C.c_int
cairo.cairo_image_surface_create.argtypes=[C.c_int,C.c_int,C.c_int]
cairo.cairo_image_surface_create.restype=C.c_void_p
cairo.cairo_create.argtypes=[C.c_void_p];cairo.cairo_create.restype=C.c_void_p
cairo.cairo_scale.argtypes=[C.c_void_p,C.c_double,C.c_double]
cairo.cairo_surface_write_to_png.argtypes=[C.c_void_p,C.c_char_p]
cairo.cairo_surface_write_to_png.restype=C.c_int
cairo.cairo_destroy.argtypes=[C.c_void_p]
cairo.cairo_surface_destroy.argtypes=[C.c_void_p]
gobject.g_object_unref.argtypes=[C.c_void_p]

def render(src,dst,scale=1.5):
    b=src.read_bytes();err=C.c_void_p()
    handle=rsvg.rsvg_handle_new_from_data(b,len(b),C.byref(err))
    if not handle:raise RuntimeError('SVG parse failure: '+str(src))
    surface=cairo.cairo_image_surface_create(0,int(1280*scale),int(720*scale))
    context=cairo.cairo_create(surface)
    cairo.cairo_scale(context,scale,scale)
    assert rsvg.rsvg_handle_render_cairo(handle,context)
    assert cairo.cairo_surface_write_to_png(surface,str(dst).encode())==0
    cairo.cairo_destroy(context);cairo.cairo_surface_destroy(surface);gobject.g_object_unref(handle)

def contact(paths,dst):
    col=3;tw=480;th=270;pad=22;label=30
    rows=(len(paths)+col-1)//col
    im=Image.new('RGB',(col*(tw+pad)+pad,rows*(th+label+pad)+pad),'#eaf0f4')
    d=ImageDraw.Draw(im);f=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18,index=2)
    for i,p in enumerate(paths):
        x=pad+(i%col)*(tw+pad);y=pad+(i//col)*(th+label+pad)
        tile=Image.open(p).convert('RGB');tile.thumbnail((tw,th),Image.Resampling.LANCZOS)
        im.paste(tile,(x,y));d.text((x+4,y+th+3),f'{i+1:02d}',fill='#172b3a',font=f)
    im.save(dst)

if __name__=='__main__':
    target=ROOT/'work/svg_render';target.mkdir(exist_ok=True)
    for src in sorted((ROOT/'svg').glob('*.svg')):render(src,target/(src.stem+'.png'))
    contact(sorted(target.glob('*.png')),ROOT/'work/SVG总览.png')
    print('SVGs rendered:',len(list(target.glob('*.png'))))
