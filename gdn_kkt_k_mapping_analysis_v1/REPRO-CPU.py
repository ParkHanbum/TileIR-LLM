from pathlib import Path
import re,json,struct,hashlib,collections
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111');D=P/'gdn_kkt_k_mapping_analysis_v1';D.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
native=P/'gdn_chunk_runtime_capture_v1/cache_v1/triton/YIMJXH566GTHAMPF7MOPC4YJK57VI7G4VRKOH25XPTIQY7JQAS7A/chunk_scaled_dot_kkt_fwd_kernel.ptx';nt=native.read_text();p=P/'gdn_chunk_kkt_qualification_v3/gdnKKTQual03/tiletmp/kktoo8sa_ro.cubin';z=p.read_bytes();off=struct.unpack_from('<Q',z,40)[0];sz,n,si=struct.unpack_from('<HHH',z,58);hs=[struct.unpack_from('<IIQQQQIIQQ',z,off+i*sz) for i in range(n)];st=hs[si];ns=z[st[4]:st[4]+st[5]]
for h in hs:
 if ns[h[0]:].split(b'\0',1)[0]==b'.nv_debug_ptx_txt':blob=z[h[4]:h[4]+h[5]];break
(D/'own-k8-embedded.ptx').write_bytes(blob);ot=blob.decode(errors='replace').replace('\0','\n')
# Native cp.async stores eight BF16 elements per thread, 256 threads.
sm={}
for tid in range(256):
 base=((tid<<4)&4080)^((tid<<1)&112)
 for plus,rowoff in [(0,0),(4096,32)]:
  for j in range(8):
   addr=base+plus+j*2;coord=(tid//8+rowoff,(tid%8)*8+j);assert addr not in sm;sm[addr]=coord
assert len(sm)==4096
r10=lambda t:(((t&7)<<7)|((t&96)<<5))|(((t&7)<<4)^((t<<1)&48))
r13=lambda t:(((t<<7)&1920)|((t<<4)&2160))^(t&16)
base_fns=[r10,lambda t:r10(t)^64,r13,lambda t:r13(t)^32,lambda t:r13(t)^64,lambda t:r13(t)^96]
# Each base is followed by a second x4 load at +4096; native split regs rs1..rs96.
def rs(tid,num):
 group=(num-1)//16;within=(num-1)%16;plus=4096 if within>=8 else 0;local=within%8;matrix=local//2;half=local%2;lane=tid%32;provider=(tid//32)*32+matrix*8+lane//4
 address=base_fns[group](provider)+plus+(lane%4)*4+half*2
 return sm[address]
cv={dst:int(src) for dst,src in re.findall(r'cvt.f32.bf16\s+(%r\d+),\s*%rs(\d+);',nt)}
regcoords=cv.copy()
for dst,beta,key in re.findall(r'mul.f32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+);',nt):
 if key in cv:regcoords[dst]=cv[key]
pat=r'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\s*\{([^}]+)\},\s*\{([^}]+)\},\s*\{([^}]+)\},\s*\{([^}]+)\};'
def parse(t):return [[re.findall(r'%r\d+',g) for g in row] for row in re.findall(pat,t)]
mm=parse(nt);assert len(mm)==32;chain=[x for x in mm if x[0]==['%r426','%r427','%r428','%r429']];assert len(chain)==8
native_groups=[];lane_details=[]
for op in chain:
 # MMA A rows0/8 live in pairs reg0/reg1; reg2/reg3 contributes second K half.
 coords=[[rs(t,regcoords[r]) for r in op[1]] for t in range(32)]
 kvals=sorted({coords[t][i][1] for t in range(4) for i in [0,2]});assert len(kvals)==8
 native_groups.append(kvals);lane_details.append(coords)
expected=[list(range(base+parity,base+16,2)) for base in range(0,64,16) for parity in (0,1)];assert native_groups==expected
full=native_groups+[[k+64 for k in g] for g in native_groups];assert sorted(sum(full,[]))==list(range(128))
# Own first MMA A from rs1,rs2,rs3,rs4 loaded at r113+0,+128,+8,+136.
# Own cp.async: row=tid//2, K=4*(tid%2), 4 BF16 values; shared row stride16B.
om={}
for tid in range(128):
 row=tid//2;k0=(tid%2)*4
 for j in range(4):om[57344+row*16+k0*2+j*2]=(row,k0+j)
def owncoords(t):
 base=57344+2*((((t>>2)&7)<<3)|(t&3)|(((t>>5)&1)<<7))
 return [om[base+d] for d in (0,128,8,136)]
own_first=[owncoords(t) for t in range(32)];ownK=sorted({own_first[t][i][1] for t in range(4) for i in (0,2)});assert ownK==list(range(8))
omma=parse(ot);assert len(omma)==96
assert omma[0][1]==['%r148','%r153','%r172','%r177']
# SSA dependency depth: each zero initializer root traverses 16 MMAs to a final tuple.
depths={};uses=collections.Counter()
for dest,A,B,C in omma:
 dt=tuple(dest);ct=tuple(C)
 if ct in depths:depth=depths[ct]+1;uses[ct]+=1
 else:depth=1
 depths[dt]=depth
leaves=[d for d in depths if not uses[d]];assert len(leaves)==6 and all(depths[d]==16 for d in leaves)
r=dict(native_ptx=str(native),native_sha256=sha(native),own_cubin=str(p),own_cubin_sha256=sha(p),native_first_8_mma_K=native_groups,native_full_16_mma_K=full,native_first_chain_lane_operand_coordinates=lane_details,own_first_mma_A_registers=omma[0][1],own_first_mma_lane_coordinates=own_first,own_first_mma_K=ownK,own_source_16_groups=[list(range(i*8,i*8+8)) for i in range(16)],own_ssa_leaf_depths=[depths[d] for d in leaves],native_vs_own_first_MMA_different=True,limitations='CPU address reconstruction assumes documented ldmatrix x4 nontrans b16 lane mapping; no hardware internals or numerical causality proof. Native BK64 loop adds64 on second iteration. Own later groups follow explicit source K8 loop; only first native/own MMA operand addresses independently decoded here.')
(D/'MAPPING.json').write_text(json.dumps(r,indent=2)+'\n');(D/'REPORT.md').write_text('Public native shared-address + ldmatrix mapping establishes first MMA K even0..14, then odd1..15, repeated per16 throughout128. Own generated first MMA A registers148/153/172/177 resolve to K0..7; source thereafter K8..15 etc. Both use same TF32 instruction/+0 accumulator, and own SSA leaves each have16 dependent MMA. This is a concrete operand grouping/order difference, not merely instruction counts or an unspecified TF32 mode. Proposal: change source col only to (block//2)*16+(block%2)+2*arange8; retain beta,casts,geometry,exp,mask. Raw success/causal conclusion requires actual run. Documented ldmatrix semantics reference https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix . CPU symbolic reconstruction is not a GPU trace. Previous mask/BK/geometry failures remain unchanged.\n')
(D/'SEAL.json').write_text(json.dumps({str(f):sha(f) for f in [native,p,*D.glob('*')]},indent=2)+'\n');print('native',native_groups);print('own',ownK,'leaves',[depths[d] for d in leaves]);print('seal',sha(D/'SEAL.json'))
