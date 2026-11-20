"""80-decimal scalar/2-state anchors; independent mpmath arithmetic."""
import mpmath as mp

def anchors(horizons,betas):
    mp.mp.dps=80;rows=[]
    def value(eta,N,beta):
        b=mp.mpf(str(beta));q=1-b;A=mp.matrix([[1-q*eta,-b*eta],[q,b]]);B=mp.matrix([-q*eta,q]);Q=B*B.T;W=mp.matrix([[mp.mpf('.5'),0],[0,0]]);S=mp.matrix([[1,0],[0,0]]);zero=mp.zeros(2);I=mp.eye(2)
        def tr(M):return M[0,0]+M[1,1]
        def compose(a,b):
            T,R,U,c=a;V,Z,F,d=b;return V*T,V*R*V.T+Z,U+T.T*F*T,c+d+tr(F*R)
        out=(I,zero,zero,mp.mpf(0));base=(A,Q,W,mp.mpf(0));n=N
        while n:
            if n&1:out=compose(out,base)
            n//=2
            if n:base=compose(base,base)
        return tr(out[2]*S)+out[3]
    for beta in betas:
      for N in horizons:
        leading=1/mp.sqrt(N);kap=mp.mpf(str((1-beta)/(1+beta)));corrected=leading-kap/(2*N)
        root=mp.findroot(lambda x:mp.diff(lambda z:value(z,N,beta),x),(corrected*mp.mpf('.95'),corrected*mp.mpf('1.05')),tol=mp.mpf('1e-65'))
        rows.append(dict(N=N,beta=beta,fixture='h=C=Sigma=D=1; zero initial momentum',eta_80=str(root),loss_80=str(value(root,N,beta)),eta_leading=str(leading),eta_corrected=str(corrected),scaled_corrected_error=str(mp.mpf(N)**mp.mpf('1.5')*(root-corrected)),status='descriptive'))
    return rows
