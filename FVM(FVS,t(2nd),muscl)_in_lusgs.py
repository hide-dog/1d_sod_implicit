import numpy
import os
import copy
import matplotlib.pyplot as pyplot

# --------------------------
# -- initial value        --
# --------------------------
nstep = 300                  # 時間ステップ数
nx0 = 100                    # 空間ステップ数
dt = 0.002                   # 時間刻み幅
dx = 0.01                    # 空間刻み幅

lbound = 1                     # 仮想境界セル数
nx = nx0+2*lbound              # 総空間セル数

# -- slope limiter --

k_muscl=1/3                        # muscl精度
b_muscl=(3-k_muscl)/(1-k_muscl)

# --定数--
gamma=1.4                    # 比熱比

norm_ok=1.0e-6

# -- 出力--
dir_name="FVS_lusgs_py_t0.01" # 出力フォルダ名
dir_name="FVS_lusgs_py" # 出力フォルダ名
out_name_front="time"                 # 出力ファイル名（先頭）
out_name_back="d-3"
# --------------------------
# -- function             --
# --------------------------
def setup():  # 初期値入力
    global x, bol, bor, qf, Qc
    u = [0.0]*nx               # 速度
    rho = [0.0]*nx             # 密度
    p = [0.0]*nx               # 圧力
    e = [0.0]*nx               # エネルギー
    x = [0.0]*nx               # 位置

    """
    e=p/(r-1)+rho*u^2/2
    """
    for i in range(nx):
        u[i] = 0.0

        if i <= nx*0.5:
            rho[i] = 1.0
            p[i] = 1.0

        else:
            rho[i] = 0.125
            p[i] = 0.1

        e[i] = p[i]/(gamma-1)+rho[i]*(u[i]**2)/2
        x[i] = i*dx-dx/2


    bol = [0.0]*3             # 左端仮想セル
    bor = [0.0]*3             # 右端仮想セル
    for j in range(3):
        if j == 0:
            bol[j] = rho[0]
            bor[j] = rho[nx-1]
        elif j == 1:
            bol[j] = u[0]*rho[0]
            bor[j] = u[nx-1]*rho[nx-1]
        elif j == 2:
            bol[j] = e[0]
            bor[j] = e[nx-1]

    qf = [[0.0] * 3 for i in [1] * nx]  # 基本量
    Qc = [[0.0] * 3 for i in [1] * nx]  # 保存量
    for i in range(nx):
        for j in range(3):
            if j == 0:
                qf[i][j] = u[i]
                Qc[i][j] = rho[i]
            elif j == 1:
                qf[i][j] = rho[i]
                Qc[i][j] = u[i]*rho[i]
            elif j == 2:
                qf[i][j] = p[i]
                Qc[i][j] = e[i]


def cal_Q():
    global Qc
    """

    cal_Res(Qc)               # 右辺Rの計算

    cal_for_lusgs()

    tempQc = copy.deepcopy(Qc)
    Qc=GS(Qc)
    
    #print("Qc")
    #print(Qc)
    """

    Qc=inner_ite2(Qc)
    #Qc = ex_inner_ite(Qc)
    
    #print("Qc")
    #print(Qc)


def bound(lQc):  # 境界の計算
    for i in range(3):
        lQc[0][i] = 2*bol[i]-lQc[1][i]  # 左端境界の計算
        lQc[nx-1][i] = lQc[nx-2][i]     # 右端の計算

    return lQc

def cal_for_lusgs(lQc,lqf):
    global Amatrix_plus,Amatrix_minus,beta_sigma
    Amatrix_plus = numpy.empty((nx,3,3))
    Amatrix_minus = numpy.empty((nx,3,3))
    beta_sigma = numpy.empty(nx)
    beta = 1.1

    I = numpy.empty((3,3))
    I[:][:] = 0.0
    for i in range(3):
        I[i][i] = 1.0


    for i in range(0, nx):
        H = (lQc[i][2]+lqf[i][2])/lQc[i][0]
        u = lqf[i][0]
        c = numpy.sqrt((gamma-1)*(H-0.5*u**2))

        sigma = abs(u) + c
        # iセルにおけるR,R^-1,Λ,|Λ|
        R, R_inv, Gam, Gam_abs = A_pm(lQc[i],lqf[i])
        A_matrix = numpy.dot((numpy.dot(R, Gam)), R_inv)

        temp = I*beta*sigma
    
        Amatrix_plus[i] = 0.5*(A_matrix+temp)
        Amatrix_minus[i] = 0.5*(A_matrix-temp)
        beta_sigma[i] = beta*sigma
    

def inner_ite2(lQc): # 内部反復

    delta_Q = numpy.array([[0.0] * 3 for i in [1] * nx])
    delta_Q_temp = numpy.array([[0.0] * 3 for i in [1] * nx])

    Qcn = numpy.array(copy.deepcopy(lQc))
    Qcm = numpy.array(copy.deepcopy(lQc))

    for ttt in range(10):

        qfm=Qctoqf(Qcm)
        
        cal_Res(Qcm)

        cal_for_lusgs(Qcm,qfm)



        delta_Q = numpy.array([[0.0] * 3 for i in [1] * nx])
        delta_Q2 = numpy.array([[0.0] * 3 for i in [1] * nx])
        delta_Q_temp = numpy.array([[0.0] * 3 for i in [1] * nx])

        lo_R=numpy.array(Res)

        sum_b=numpy.array([0.0] * 3)
        for i in range(lbound,nx-lbound):
                sum_b += abs(lo_R[i])

        ite=0
        con=0
        while con==0:
            delta_Q_temp = copy.deepcopy(delta_Q)

            L = numpy.array([[0.0] * 3 for i in [1] * nx])
            D = numpy.array([0.0]*nx)
            U = numpy.array([[0.0] * 3 for i in [1] * nx])
            for i in range(nx):
                L[i] = dt*(numpy.dot(Amatrix_plus[i],delta_Q[i]))/2
                D[i] = dx+dx+dt*(beta_sigma[i])
                U[i] = dt*(numpy.dot(Amatrix_minus[i],delta_Q[i]))/2
            
            # jacobi??
            
            
            
            RHS = numpy.array([[0.0] * 3 for i in [1] * nx])
            # RHS
            for i in range(lbound,nx-lbound):
                RHS[i] = -(Qcm[i]-Qcn[i])*dx-dt*lo_R[i]

            # (D+L)Q=Q
            for i in range(lbound,nx-lbound):
                delta_Q[i] = (L[i-1]+RHS[i]) / D[i]

            # (D+U)Q=DQ
            for i in range(lbound,nx-lbound):
                delta_Q[i] = delta_Q[i] - U[i+1]/ D[i]

            #print("xxxxxxx")

            #print(delta_Q[2])
            #print(delta_Q_temp[2])
                

            #for i in range(lbound,nx-lbound):
                #delta_Q2[i]=dt/dx*(numpy.dot(yacobiA[i-1],delta_Q2[i-1])-numpy.dot(yacobiA[i+1],delta_Q2[i+1]))/2-dt/dx*lo_R[i]

            if (ite+1) % 100 ==0:
                sum_b_Ax=numpy.array([0.0] * 3)

                
                """
                for i in range(nx):
                    L[i] = dt*(numpy.dot(yacobiA[i],delta_Q[i]))/2
                    D[i] = dx
                    U[i] = dt*(numpy.dot(yacobiA[i],delta_Q[i]))/2
                # (D+L)Q=Q
                for i in range(lbound,nx-lbound):
                    delta_Q_temp[i] = (L[i-1]+RHS[i]) / D[i]
                # (D+U)Q=DQ
                for i in range(lbound,nx-lbound):
                    delta_Q_temp[i] = delta_Q_temp[i] - U[i+1]/ D[i]
                """
                    

                for i in range(lbound,nx-lbound):
                    for j in range(3):
                        #sum_b_Ax += abs(-delta_Q[i]+dt/dx*(numpy.dot(yacobiA[i-1],delta_Q[i-1])-numpy.dot(yacobiA[i+1],delta_Q[i+1]))/2-dt/dx*lo_R[i])
                        sum_b_Ax += abs(delta_Q[i]-delta_Q_temp[i])
                

                norm2d=[0.0] * 3

                for i in range(3):
                    norm2d[i]=sum_b_Ax[i]/sum_b[i]

                print(norm2d)

                if norm2d[0] < norm_ok and norm2d[1] < norm_ok and norm2d[2] < norm_ok:
                    con=1

            ite += 1

        delta_Q.tolist()

        """
        print("dQ")
        print(delta_Q)
        print("Qcm")
        print(Qcm)"""

            

        for i in range(lbound,nx-lbound):
            for j in range(3):
                Qcm[i][j]=Qcm[i][j]+delta_Q[i][j]

        
        #print(Qcm)
        #raise ValueError("error!")

        """
        print("Qcm")
        print(Qcm)
        """

        Qcm=bound(Qcm)
    

        
    return Qcm


def ex_inner_ite(lQc):
    delta_Q = numpy.array([[0.0] * 3 for i in [1] * nx])
    delta_Q_temp = numpy.array([[0.0] * 3 for i in [1] * nx])

    Qcn = numpy.array(copy.deepcopy(lQc))
    Qcm = numpy.array(copy.deepcopy(lQc))

    for ttt in range(10):
        qfm = Qctoqf(Qcm)

        cal_Res(Qcm)
        lo_R=numpy.array(Res)

        for i in range(1,nx-1):
            Qcm[i] = Qcn[i]-dt/dx*lo_R[i]
        
    return Qcm

def inv_matrix(D):
    # Sweep method
    for l in range(3):
        aa       = 1.0 / D[l][l]
        D[l][l] = 1.0

        for m in range(3):
            D[l][m] = D[l][m] * aa

        for m in range(3):
            if m != l:
                bb       = D[m][l]
                D[m][l] = 0.0
                for n in range(3):
                    D[m][n] = D[m][n] - bb*D[l][n]
                
    return D


def cal_Res(lQc):  # 境界フラックスの計算
    global Res

    Res = numpy.array([[0.0] * 3 for i in [1] * nx])
    fvs(lQc)                             # FVS法によるフラックスの作成

    for i in range(1, nx-1):
        Res[i] = Fplus[i]-Fplus[i-1]

    Res.tolist()


def fvs(lQc):  # FVS法によるフラックスの計算(セル1と2の境界をFplus[1]に格納)
    global Fplus

    Fplus = [[0.0] * 3 for i in [1] * (nx+1)]
    muscl(lQc)

    for i in range(0, nx-1):
        # i+1/2セルにおけるR,R^-1,Λ,|Λ|
        R, R_inv, Gam, Gam_abs = A_pm(QcL[i],qfL[i])
        Ap = numpy.dot((numpy.dot(R, Gam+Gam_abs)), R_inv)               # 固有値が正のものを計算

        # i+1/2セルにおけるR,R^-1,Λ,|Λ|
        R, R_inv, Gam, Gam_abs = A_pm(QcR[i],qfR[i])
        Am = numpy.dot((numpy.dot(R, Gam-Gam_abs)), R_inv)               # 固有値が負のものを計算

        Fplus[i] = 0.5*(numpy.dot(Ap, QcL[i]) + numpy.dot(Am, QcR[i]))   # フラックスを計算


def A_pm(lQc,lqf):  # ヤコビアン行列の固有値もろもろ計算
    H = (lQc[2]+lqf[2])/lQc[0]  # エンタルピー
    u = lqf[0]
    c = numpy.sqrt((gamma-1)*(H-0.5*u**2))
    b_para = (gamma-1)/c**2
    a_para = 0.5*b_para*u**2

    R = numpy.array([[1.0, 1.0, 1.0, ], [u-c, u, u+c],
                     [H-u*c, 0.5*u**2, H+u*c]])
    R_inv = numpy.array([[0.5*(a_para+u/c), 0.5*(-b_para*u-1/c), 0.5*b_para], [
                        1-a_para, b_para*u, -b_para], [0.5*(a_para-u/c), 0.5*(-b_para*u+1/c), 0.5*b_para]])
    Gam = numpy.array([[(u-c), 0.0, 0.0], [0.0, u, 0.0], [0.0, 0.0, (u+c)]])
    Gam_abs = numpy.array(
        [[abs(u-c), 0.0, 0.0], [0.0, abs(u), 0.0], [0.0, 0.0, abs(u+c)]])

    return R, R_inv, Gam, Gam_abs

def yacobi_A(lQc,lqf):

    yacobiAp=[0.0] * nx
    yacobiAm=[0.0] * nx
    for i in range(nx):
        R, R_inv, Gam, Gam_abs = A_pm(lQc[i],lqf[i])
        yacobiAp[i] = numpy.dot((numpy.dot(R, Gam+Gam_abs)), R_inv)
        yacobiAm[i] = numpy.dot((numpy.dot(R, Gam-Gam_abs)), R_inv)

    return yacobiAp,yacobiAm

def muscl(lQc):
    global qf,qfL,qfR,QcL,QcR
    # 1と2の間を1に収納

    lqf=Qctoqf(lQc)

    qfL=[[0.0] * 3 for i in [1] * (nx+1)]
    qfR=[[0.0] * 3 for i in [1] * (nx+1)]

    for i in range(1,nx-2):
        for j in range(3):
            dplus_j=lqf[i+1][j]-lqf[i][j]
            dminus_j=lqf[i][j]-lqf[i-1][j]
            dplus_jp=lqf[i+2][j]-lqf[i+1][j]
            dminus_jp=lqf[i+1][j]-lqf[i][j]
            
            qfL[i][j]=lqf[i][j]+1/4*((1-k_muscl)*minmod(dminus_j,dplus_j,b_muscl)+(1+k_muscl)*minmod(dplus_j,dminus_j,b_muscl))
            qfR[i][j]=lqf[i+1][j]-1/4*((1-k_muscl)*minmod(dplus_jp,dminus_jp,b_muscl)+(1+k_muscl)*minmod(dminus_jp,dplus_jp,b_muscl))

    # 境界内側用
    for j in range(3):
        dplus_jp=lqf[2][j]-lqf[1][j]
        dminus_jp=lqf[1][j]-lqf[0][j]
        qfR[0][j]=lqf[1][j]-1/4*((1-k_muscl)*minmod(dplus_jp,dminus_jp,b_muscl)+(1+k_muscl)*minmod(dminus_jp,dplus_jp,b_muscl))

        dplus_j=lqf[nx-1][j]-lqf[nx-2][j]
        dminus_j=lqf[nx-2][j]-lqf[nx-3][j]
        qfL[nx-2][j]=lqf[nx-2][j]+1/4*((1-k_muscl)*minmod(dminus_j,dplus_j,b_muscl)+(1+k_muscl)*minmod(dplus_j,dminus_j,b_muscl))
        
            
    QcL=qftoQc(qfL)
    QcR=qftoQc(qfR)

    # 境界外側用(境界は風上)
    qfL[0]=lqf[0][:]
    QcL[0]=lQc[0][:]
    qfR[nx-2]=lqf[nx-1][:]
    QcR[nx-2]=lQc[nx-1][:]
    

def minmod(x,y,b):
    ans=numpy.sign(x)*max(0,min(abs(x),numpy.sign(x)*y*b))
        
    return ans

    
def qftoQc(qf):  # 基本量から保存量変換
    lo_Qc=[[0.0] * 3 for i in [1] * nx]
    for i in range(nx):
        for j in range(3):
            if j ==0:
                lo_Qc[i][j]=qf[i][1]
            elif j ==1:
                lo_Qc[i][j]=qf[i][1]*qf[i][0]
            elif j ==2:
                lo_Qc[i][j]=(qf[i][2]/(gamma-1)+1.0/2.0*qf[i][1]*(qf[i][0]**2))

    return lo_Qc

def Qctoqf(Qc):  # 保存量から基本量変換
    lo_qf=[[0.0] * 3 for i in [1] * nx]
    for i in range(nx):
        for j in range(3):
            if j ==0:
                lo_qf[i][j]=Qc[i][1]/Qc[i][0]
            elif j ==1:
                lo_qf[i][j]=Qc[i][0]
            elif j ==2:
                lo_qf[i][j]=(gamma-1)*(Qc[i][2]-1.0/2.0*Qc[i][0]*((Qc[i][1]/Qc[i][0])**2))
    return lo_qf

    
def output_q(f_name):  # テキスト形式で出力
    outlist=["x[m] u[m/s] rho[kg/m3] p[Pa]"]  # 出力するものの名前
    for i in range(len(qf)):
        outlist.append(str(x[i])+" "+str(qf[i][0])+" "+str(qf[i][1])+" "+str(qf[i][2]))
    outlist='\n'.join(outlist)

    with open(dir_name+"/"+f_name,'wt') as f:
        f.write(outlist)

def cre_dir():  # フォルダ作成
    try:
        os.mkdir(dir_name)
    except:
        pass

# --------------------------
# -- preparetion          --
# --------------------------
cre_dir()
setup()
# --------------------------
# -- main                 --
# --------------------------

for k in range(nstep):
    
    print(k)
    cal_Q()
    qf=Qctoqf(Qc)
    
    output_q(out_name_front+'{:0=4}'.format(int(k*dt*1000))+out_name_back)
