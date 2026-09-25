"""Read flat scalar columns in the supplied Snappy/PLAIN/RLE-dictionary Parquet.
A narrow archival-conversion fallback; unsupported encodings fail explicitly.
"""
from pathlib import Path
import struct, ctypes, ctypes.util
from thrift.transport import TTransport
from thrift.protocol import TCompactProtocol
from thrift.Thrift import TType
import numpy as np

def struct_at(b, offset):
    tr=TTransport.TMemoryBuffer(b[offset:]); p=TCompactProtocol.TCompactProtocol(tr)
    def val(tp):
        if tp==TType.STRUCT:
            p.readStructBegin();d={}
            while True:
                _,t,f=p.readFieldBegin()
                if t==TType.STOP:break
                d[f]=val(t);p.readFieldEnd()
            p.readStructEnd();return d
        if tp==TType.LIST:
            t,n=p.readListBegin();a=[val(t) for _ in range(n)];p.readListEnd();return a
        if tp==TType.MAP:
            k,v,n=p.readMapBegin();d={val(k):val(v) for _ in range(n)};p.readMapEnd();return d
        return {TType.BOOL:p.readBool,TType.BYTE:p.readByte,TType.I16:p.readI16,TType.I32:p.readI32,TType.I64:p.readI64,TType.DOUBLE:p.readDouble,TType.STRING:p.readBinary}[tp]()
    out=val(TType.STRUCT);return out,tr.cstringio_buf.tell()

def snappy(b,n):
    lib=ctypes.CDLL(ctypes.util.find_library('snappy'))
    lib.snappy_uncompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
    out=ctypes.create_string_buffer(n);sz=ctypes.c_size_t(n)
    assert lib.snappy_uncompress(b,len(b),out,ctypes.byref(sz))==0
    assert sz.value==n
    return out.raw[:n]

def uvar(b,o):
    x=s=0
    while True:
        z=b[o];o+=1;x|=(z&127)<<s
        if z<128:return x,o
        s+=7

def hybrid(b, width, count):
    out=[];pos=0
    while len(out)<count:
        h,pos=uvar(b,pos)
        if h&1:
            n=(h>>1)*8;nb=(n*width+7)//8
            bits=int.from_bytes(b[pos:pos+nb],'little');pos+=nb
            mask=(1<<width)-1
            out.extend((bits>>(i*width))&mask for i in range(n))
        else:
            n=h>>1;nb=(width+7)//8
            v=int.from_bytes(b[pos:pos+nb],'little');pos+=nb
            out.extend([v]*n)
    return out[:count]

def plain(b,ty,n):
    if ty in [1,2,4,5]:
        return np.frombuffer(b,dtype={1:'<i4',2:'<i8',4:'<f4',5:'<f8'}[ty],count=n).tolist()
    if ty==0:return [(b[i//8]>>(i%8))&1 == 1 for i in range(n)]
    if ty==6:
        out=[];o=0
        for _ in range(n):
            l=struct.unpack_from('<I',b,o)[0];o+=4;out.append(b[o:o+l].decode());o+=l
        return out
    raise NotImplementedError(ty)

def read_scalar(path):
    import pandas as pd
    b=Path(path).read_bytes();assert b[:4]==b[-4:]==b'PAR1'
    ml=struct.unpack_from('<I',b,len(b)-8)[0];meta,_=struct_at(b,len(b)-8-ml)
    optional={s[4].decode():s.get(3)==1 for s in meta[2] if 1 in s}
    allrows=[]
    for rg in meta[4]:
        cols={}
        for col in rg[1]:
            cm=col[3];pathparts=cm[3]
            if len(pathparts)>1:continue
            name=pathparts[0].decode();out=[];dictionary=None
            off=min(cm.get(11,cm[9]),cm[9]);end=off+cm[7]
            while off<end and len(out)<cm[5]:
                h,k=struct_at(b,off);off+=k
                payload=b[off:off+h[3]];off+=h[3]
                if cm[4]==1:payload=snappy(payload,h[2])
                elif cm[4]!=0:raise NotImplementedError(('codec',cm[4]))
                if h[1]==2:
                    dh=h[7];assert dh[2]==0;dictionary=plain(payload,cm[1],dh[1]);continue
                if h[1]!=0:raise NotImplementedError(('page',h))
                dh=h[5];n=dh[1];encoding=dh[2];pos=0
                if optional[name]:
                    assert dh[3]==3
                    l=struct.unpack_from('<I',payload,pos)[0];pos+=4
                    defs=hybrid(payload[pos:pos+l],1,n);pos+=l
                else:defs=[1]*n
                nn=sum(defs);data=payload[pos:]
                if encoding==0:vals=plain(data,cm[1],nn)
                elif encoding in (2,8):
                    indices=hybrid(data[1:],data[0],nn);vals=[dictionary[i] for i in indices]
                elif encoding==3 and cm[1]==0:
                    l=struct.unpack_from('<I',data,0)[0];vals=[bool(x) for x in hybrid(data[4:4+l],1,nn)]
                else:raise NotImplementedError(('encoding',encoding,cm[1]))
                it=iter(vals);out.extend(next(it) if d else None for d in defs)
            assert len(out)==cm[5],(name,len(out),cm[5]);cols[name]=out
        allrows.append(pd.DataFrame(cols))
    frame=pd.concat(allrows,ignore_index=True);assert len(frame)==meta[3]
    return frame
if __name__=='__main__':
    import sys
    df=read_scalar(sys.argv[1]);df.to_csv(sys.argv[2],index=False,float_format='%.17g')
    print('Converted',df.shape, 'to',sys.argv[2])
