import sys,time; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
from PIL import Image
from dressup.engine import Engine
I='/workspace/dress-up-trial/inputs/'; O='/workspace/dress-up-windows-trial/'
q=sys.argv[1]; jobs=[a.split(':') for a in sys.argv[2:]]
e=Engine('/workspace/dress-up-windows-work/models'); t=time.time(); e.load(); print('load',round(time.time()-t,1),flush=True)
p=Image.open(I+'person_dress.jpg')
for name,g,kind in jobs:
    d={}; t=time.time()
    r=e.try_on(p,Image.open(I+g),kind,q,progress=lambda i,n:print(i,n,round(time.time()-t),flush=True) if i%5==0 else None,debug=d)
    r.save(O+f'result_{name}_{q}.png'); d['mask'].save(O+f'debug_mask_{name}.png'); d['gen'].save(O+f'debug_gen_{name}_{q}.png')
    print(name,'secs',round(d['secs'],1),flush=True)
