"""Physical ft/psi/day IMPES data for section 6 of Ye et al. (2025)."""
from pathlib import Path
import hashlib
import json
import numpy as np
from spe10_problem import REV, SHAPE, DX, DY, DZ

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results'/'spe10_model2_dynamic'
FT3_PER_BBL = .158987294928/.028316846592
FLOW_FACTOR = 9.869233e-16*.3048*6894.757293168/.001*86400/.028316846592
BHP = 4000.
INJECTION = 5000.*FT3_PER_BBL


def prepare():
    OUT.mkdir(exist_ok=True)
    src = OUT/'SPE10MODEL2_PHI.INC'
    if not src.exists():
        from urllib.request import urlretrieve
        urlretrieve(f'https://raw.githubusercontent.com/OPM/opm-data/{REV}/spe10model2/{src.name}',src)
    raw = src.read_bytes()
    body = raw.split(b'PORO',1)[1].strip().split(b'/')[0]
    porosity = np.fromstring(body,sep=' ').reshape(SHAPE)
    if not np.isfinite(porosity).all() or np.min(porosity)<0 or np.max(porosity)>1:
        raise ValueError('invalid porosity')
    np.save(OUT/'porosity.npy',np.maximum(porosity,.05))
    meta = dict(url=f'https://raw.githubusercontent.com/OPM/opm-data/{REV}/spe10model2/{src.name}',
                raw_sha256=hashlib.sha256(raw).hexdigest(),minimum_porosity=.05,
                native_range=[float(porosity.min()),float(porosity.max())],
                flow_factor_ft3_day_per_md_ft_psi_cp=FLOW_FACTOR,ft3_per_bbl=FT3_PER_BBL,
                bhp_psi=BHP,injection_ft3_day=INJECTION)
    (OUT/'data.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta


def load():
    k = np.load(ROOT/'results'/'spe10_model2_singlephase'/'permeability.npy')
    phi = np.load(OUT/'porosity.npy')
    if k.shape!=(3,*SHAPE) or phi.shape!=SHAPE or not np.isfinite(k).all() or not np.isfinite(phi).all() or np.min(k)<=0 or np.min(phi)<.05:
        raise ValueError('invalid complete SPE10 fields')
    return k,phi


class Problem:
    def __init__(self,k,phi):
        self.k,self.shape = k,tuple(k.shape[1:])
        self.pv = (np.asarray(phi)*DX*DY*DZ).ravel()
        self.ids = np.arange(phi.size,dtype=np.int32).reshape(phi.shape)
        self.inj = np.zeros(phi.size)
        nz,ny,nx = phi.shape
        self.inj[self.ids[:,(ny-1)//2,(nx-1)//2].ravel()] = INJECTION/nz
        producer = np.zeros(phi.shape)
        for y in (0,ny-1):
            for x in (0,nx-1):
                producer[:,y,x] = 1
        r1 = np.sqrt(9*DX**2+DY**2)/2
        r2 = np.sqrt(DX**2+9*DY**2)/2
        wi = (DX**-2+DY**-2)/((2/np.pi)*((DY/DX)*np.log(r1)+(DX/DY)*np.log(r2))-1)
        self.well_base = (producer*k[0]*wi*DX*DY*DZ*FLOW_FACTOR).ravel()
        self.ends,self.half_resistance = [],[]
        for axis,(part,geom) in enumerate(((k[2],DX*DY/DZ),(k[1],DX*DZ/DY),(k[0],DY*DZ/DX))):
            a,b = [slice(None)]*3,[slice(None)]*3
            a[axis],b[axis] = slice(None,-1),slice(1,None)
            a,b = tuple(a),tuple(b)
            self.ends.append((self.ids[a].ravel(),self.ids[b].ravel()))
            self.half_resistance.append((1/(2*geom*FLOW_FACTOR*part[a].ravel()),
                                         1/(2*geom*FLOW_FACTOR*part[b].ravel())))
        self.i = np.concatenate([v[0] for v in self.ends])
        self.j = np.concatenate([v[1] for v in self.ends])

    def pressure_system(self,s):
        from spe10_dynamic_transport import phase_properties
        lam,_ = phase_properties(s.ravel())
        weights=[]
        for (i,j),(ri,rj) in zip(self.ends,self.half_resistance):
            weights.append(1/(ri/lam[i]+rj/lam[j]))
        tag = self.k[0]*lam.reshape(self.shape)
        return tag,(self.i,self.j,np.concatenate(weights)),self.well_base*lam,self.inj


if __name__=='__main__':
    print(json.dumps(prepare(),indent=2))
