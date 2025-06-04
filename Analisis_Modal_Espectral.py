# Análisis Modal Espectral
# Pórtico de Concreto Reforzado de 5 niveles
# Autor: Victor Reino - Juan David Ruiz
# Director: John Esteban Ardila

# Unidades
# Longitud: m
# Masa:     kg
# Tiempo:   s
# Fuerza:   N = kg.m/s^2
# Esfuerzo: Pa = N/m^2 (módulo de elasticidad)

# Importar librerías
import openseespy.opensees as ops
import opsvis as opsv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# Parámetros a modificar:
# Incluir secciones: 0-gruesas, 1-fisuradas (Ic = 0.8 Igc, Iv = 0.5 Igv)
IncSF = 0
# Incluir la mampostería: 0-No, 1:-Si
IncM = 0
# Considerar carga viva: 0:No y 1:Si
consL = 0
# Sección de columnas y vigas [b, h] en cm
scol = [50.0, 50.0]
svig = [50.0, 50.0]
# Número de pisos
Npisos = 5
# Amenaza sísmica
Aa, Av, I, TS = 0.15, 0.15, 1.0, 'E' # Montería - Córdoba
# FA: Factor de Amplificación para (0.8Vs)/Vst > 1.0
FAx, FAy = 1.0, 1.0
# Modos
Nmodes = 6 # no. modos de vibración a calcular
Nmode = 6 # no. modos de vibración a graficar

# Datos de entrada
g = 9.81 # acc. gravedad (m/s^2)

# Datos de entrada:
# Geometria
LVx, nVx = 5.0, 3 # longitud en x (m), número de vanos en x
LVy, nVy = 5.0, 3 # longitud en y (m), número de vanos en y
dHz, nPz = 3.0, Npisos # altura en z (m), número de pisos en z
Lx = np.ones(nVx)*LVx
Ly = np.ones(nVy)*LVy
Hz = np.ones(nPz)*dHz
n_col = nPz * (nVx + 1) * (nVy + 1)
n_vigx = nVx * (nVy + 1) * nPz
n_vigy = nVy * (nVx + 1) * nPz
n_elem = n_col+n_vigx+n_vigy
ndpi = (nVx+1)*(nVy+1) # no. de nodos por piso

# Crear matrices de coordenadas (nodos) y elementos
def generar_estructura(Lx, Ly, Hz, graficar=True):
    # Datos de entrada:
    # Lx: vector de longitudes de los vanos en x
    # Ly: vector de longitudes de los vanos en y
    # Hz: vector de alturas de los pisos en z
    # Datos de salida:
    # nodos: matriz de coordenadas de los nodos
    # elementos: matriz de conformación de los elementos
    # losas: matriz de las losas del piso tipo
    
    nVx, nVy, nPz = len(Lx), len(Ly), len(Hz)

    # Calculo de nodos
    n_nodos = (nVx + 1) * (nVy + 1) * (nPz + 1)
    nodos = np.zeros((n_nodos, 3), 'f8')

    cont_nod = 0
    for k in range(nPz + 1):
        for j in range(nVy + 1):
            for i in range(nVx + 1):
                nx = sum(Lx[:i]) if i > 0 else 0.0
                ny = sum(Ly[:j]) if j > 0 else 0.0
                nz = sum(Hz[:k]) if k > 0 else 0.0
                nodos[cont_nod, :] = [nx, ny, nz]
                cont_nod += 1

    # Crear lista de elementos
    n_col = nPz * (nVx + 1) * (nVy + 1)
    n_vigx = nVx * (nVy + 1) * nPz
    n_vigy = nVy * (nVx + 1) * nPz
    n_elementos = n_col + n_vigx + n_vigy
    elementos = np.zeros((n_elementos, 2), 'int')

    # Columnas (avanzar en z, luego en x, luego en y)
    col_idx = 0
    for j in range(nVy + 1):  # Fila en y
        for i in range(nVx + 1):  # Columna en x
            for k in range(nPz):  # Piso en z
                nodo_inicial = k * (nVx + 1) * (nVy + 1) + j * (nVx + 1) + i
                nodo_final = nodo_inicial + (nVx + 1) * (nVy + 1)
                elementos[col_idx] = (nodo_inicial + 1, nodo_final + 1)
                col_idx += 1

    # Vigas en x (avanzar en x, luego en z, luego en y)
    vigx_idx = col_idx
    for k in range(1, nPz + 1):  # Solo para pisos superiores
        for j in range(nVy + 1):
            for i in range(nVx):
                nodo_inicial = k * (nVx + 1) * (nVy + 1) + j * (nVx + 1) + i
                nodo_final = nodo_inicial + 1
                elementos[vigx_idx] = (nodo_inicial + 1, nodo_final + 1)
                vigx_idx += 1

    # Vigas en y (avanzar en y, luego en x, luego en z)
    vigy_idx = vigx_idx
    for k in range(1, nPz + 1):  # Solo para pisos superiores
        for i in range(nVx + 1):
            for j in range(nVy):
                nodo_inicial = k * (nVx + 1) * (nVy + 1) + j * (nVx + 1) + i
                nodo_final = nodo_inicial + (nVx + 1)
                elementos[vigy_idx] = (nodo_inicial + 1, nodo_final + 1)
                vigy_idx += 1
    
    # Crear las losas en el nivel z = 0 usando np.array directamente
    losas = np.zeros((nVx * nVy, 4), 'int')  # Inicializamos el array con ceros
    for i in range(nVx):  # Recorrer en x
        for j in range(nVy):  # Recorrer en y
            nodo1 = i + j * (nVx + 1) + 1  # Nodo 1
            nodo2 = (i + 1) + j * (nVx + 1) + 1  # Nodo 2
            nodo3 = (i + 1) + (j + 1) * (nVx + 1) + 1  # Nodo 3
            nodo4 = i + (j + 1) * (nVx + 1) + 1  # Nodo 4

            # Asignar los valores directamente a la matriz 'losas'
            losas[i + j * nVx] = [nodo1, nodo2, nodo3, nodo4]

    return nodos, elementos, losas

nodos, elementos, losas = generar_estructura(Lx, Ly, Hz)

# # Importar geometría desde el archivo de excel
# archivo = 'Datos.xlsx'
# nodos = pd.read_excel(archivo, sheet_name = 'Nodos', usecols = [0,1,2]).to_numpy(dtype = np.float64)
# elementos = pd.read_excel(archivo, sheet_name = 'Elementos', usecols = [1,2]).to_numpy(dtype = np.int32)
# losas = pd.read_excel(archivo, sheet_name = 'Losas', usecols = [0,1,2,3]).to_numpy(dtype = np.int32)

# Materiales:
# Concreto
fc = 21.0e6 # resistencia especificada a la compresión (Pa)
Ec = 4700*(fc*1e-6)**0.5*1e6 # módulo de elasticidad (Pa)
vc = 0.2 # coeficiente de Poisson
Gc = Ec/(2*(1+vc)) # módulo de corte (Pa)
fct = 0.62*(fc*1e-6)**0.5*1e6 # resistencia a la tracción
gammac = 24.0e3 # peso específico (N/m^3)
# Acero
fy = 420.0e6 # esfuerzo de fluencia (Pa)
Es = 200.0e9 # módulo de elasticidad (Pa)
# Mampostería
fm = 5e6 # resistencia a la compresión bloques de concreto
Em = 900*fm if 900*fm*1e-9 <= 20 else 20e9 # módulo de elasticidad de los bloques

# Secciones transversales:
# Columnas
bc, hc, rkc = scol[0]*1e-2, scol[1]*1e-2, 1.00 if IncSF == 0 else 0.80
Ac , I1c, I2c = bc*hc, 1/12*bc*hc**3*rkc, 1/12*bc**3*hc*rkc
ac, cc = max(bc, hc), min(bc, hc)
betac = 1/3-0.21*cc/ac*(1-(cc/ac)**4/12)
J3c = betac*cc**3*ac
mc = Ac*gammac/g

# Vigas
bv, hv, rkv = svig[0]*1e-2, svig[1]*1e-2, 1.00 if IncSF == 0 else 0.50
Av , I1v, I2v = bv*hv, 1/12*bv*hv**3*rkv, 1/12*bv**3*hv*rkv
av, cv = max(bv, hv), min(bv, hv)
betav = 1/3-0.21*cv/av*(1-(cv/av)**4/12)
J3v = betav*cv**3*av
mv = Av*gammac/g

# Mampostería: diagonal equivalente
tm = 10e-2
theta = np.arctan(dHz/Lx[0])
hm = dHz-hv
lm = Lx[0]-hc
lambda1 = ((Em*tm*np.sin(2*theta))/(4*Ec*I1c*hm))**(1/4)
Dm = (lm**2+hm**2)**0.5
am = 0.175*Dm*(lambda1*dHz)**-0.4
Am = am*tm

# Cargas: Muerta y Viva
wle = 2.37e3 # carga de entrepiso (N/m^2)
wfp = 3.00e3 # carga fachadas y particiones (N/m^2)
wap = 1.60e3 # carga acabados y afinado de piso (N/m^2)
wL = 1.80e3 # carga viva (N/m^2)
wT = 1.00*(wle+wfp+wap)+0.25*wL*consL

# Definición del modelo
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# Definir los nodos
for i, nodoi in enumerate(nodos):
    ops.node(i+1, *nodoi)

# Apoyos: restricciones en el nivel z = 0.0 m
ops.fixZ(0.0, *[1, 1, 1, 1, 1, 1])

# Definición de elementos
CTransform = 1
BTransformx = 2
BTransformy = 3

ops.geomTransf('PDelta', CTransform, *[1, 0, 0])
ops.geomTransf('Linear', BTransformx, *[0, -1, 0])
ops.geomTransf('Linear', BTransformy, *[1, 0, 0])

# Ensamble de columnas y vigas
for i, elem in enumerate(elementos):
    nodoi = int(elem[0]) # nodo inicial
    nodoj = int(elem[1]) # nodo final
    if i <= n_col-1:
        # Columnas
        ops.element('elasticBeamColumn', i+1, nodoi, nodoj,
                    Ac, Ec, Gc, J3c, I1c, I2c, CTransform, '-mass', mc)
    else:
        if i <= (n_col + n_vigx)-1:
            # Vigas en x
            ops.element('elasticBeamColumn', i+1, nodoi, nodoj,
                        Av, Ec, Gc, J3v, I1v, I2v, BTransformx, '-mass', mv)
        else:
            # Vigas en y
            ops.element('elasticBeamColumn', i+1, nodoi, nodoj,
                        Av, Ec, Gc, J3v, I1v, I2v, BTransformy, '-mass', mv)

# Incluir muros
if IncM == 1:
    # Definir el material: mampostería
    Tagmat = 1
    ops.uniaxialMaterial('Elastic', Tagmat, Em)
    nMuros = 4*nPz
    nodin = np.array([3, 15, 9, 12], dtype='int') # nodos iniciales
    cont = 0
    for i in range(n_elem+1, n_elem+1+nMuros):
        print(i)
        if cont <= nPz-1:
            nodoin = nodin[0]+cont*ndpi
            nodofn = nodoin+(ndpi-1)
        else:
            if cont <= 2*nPz-1:
                nodoin = nodin[1]+(cont-nPz)*ndpi
                nodofn = nodoin+(ndpi-1)
            else:
                if cont <= 3*nPz-1:
                    nodoin = nodin[2]+(cont-2*nPz)*ndpi
                    nodofn = nodoin+(ndpi-4)
                else:
                    nodoin = nodin[3]+(cont-3*nPz)*ndpi
                    nodofn = nodoin+(ndpi-4)
        print(nodoin, nodofn)
        # Diagonales: Mampostería
        ops.element('Truss', i, *[int(nodoin), int(nodofn)], Am, Tagmat)
        cont += 1

# Graficar nodos y elementos
opsv.plot_model(fig_wi_he = (35.0, 40.0), fig_lbrt = (0, 0, 1, 1),
                axis_off = 0, az_el = (-140, 30), local_axes = False)
plt.savefig('NodElemP3D5N.png', dpi = 300, bbox_inches = 'tight')
plt.show()

if IncM == 0:
    # Graficar modelo extruído
    ele_shapes = {}
    for i in range(len(elementos)):
        if i <= n_col-1:
            ele_shapes[i+1] = ['rect',[bc, hc]]
        else:
            ele_shapes[i+1] = ['rect',[bv, hv]]

    opsv.plot_extruded_shapes_3d(ele_shapes, fig_wi_he = (35.0, 40.0), fig_lbrt = (0, 0, 1, 1),
                    az_el = (-140, 30))
    plt.savefig('PExtruido3D5N.png', dpi = 300, bbox_inches = 'tight')
    plt.show()


# Definición de masa concentrada en nodos
# función que obtenga centro de masa, áreas tributarias, masa/nodo
def CM_Masa_Nodo(w_total, losas, nodos_losa):
    nnd = len(nodos_losa)
    nlos = len(losas)
    areas = np.zeros(nnd, 'f8')
    
    def xyz2area(xyz):
        # Fórmula de Gauss
        x, y = xyz[:,0], xyz[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    # Calcular áreas tributarias de cada nodo
    for i in range(nlos):
        xyz = nodos_losa[losas[i]-1]
        area = xyz2area(xyz)
        area_trib = area/len(xyz)
        areas[losas[i]-1] += area_trib
    
    masa_nodos = areas*w_total/g
    masa_total = np.sum(masa_nodos)
    xcm = np.sum(nodos_losa[:,0]*masa_nodos)/masa_total
    ycm = np.sum(nodos_losa[:,1]*masa_nodos)/masa_total
    CM = [xcm, ycm]
    return CM, masa_nodos

# CM y masa/nodo para piso tipo
CM, mnd = CM_Masa_Nodo(wT, losas, nodos[0:ndpi,:])
mnd[[5, 6, 9, 10]] = 3/4 * mnd[[5, 6, 9, 10]]
# CM y masa/nodo para techo
CMt, mndt = CM_Masa_Nodo(wT-wfp, losas, nodos[0:ndpi,:])
mndt[[5, 6, 9, 10]] = 3/4 * mndt[[5, 6, 9, 10]]

# Agregar nodos CM en el modelo
# GDL de cada entrepiso
gdlcm = [0, 0, 1, 1, 1, 0]

# Crear nodos de CM para cada nivel
Hz = np.ones(nPz)*dHz # alturas de entrepiso de cada nivel
for i in range(nPz):
    Tagnode_CM = len(nodos)+i+1 # etiqueta de los nodos de CM
    if i != nPz-1:
        ops.node(Tagnode_CM, *CM, sum(Hz[:i+1]))
    else:
        ops.node(Tagnode_CM, *CMt, sum(Hz[:i+1]))
    
    ops.fix(Tagnode_CM, *gdlcm) # fijar restricciones

# Definir los diafragmas de piso (constraints)
perpDirn = 3
for i in range(nPz):
    Tagnode_CM = len(nodos)+i+1 # etiqueta de los nodos de CM
    nodoi = (ndpi+1)+i*ndpi # nodo incial cada nivel
    nodof = nodoi + ndpi # nodo final de cada nivel
    for cNodeTags in range(nodoi, nodof):
        ops.rigidDiaphragm(perpDirn, Tagnode_CM, cNodeTags)

# Asignar masas a cada nivel
for i in range(nPz):
    nodoi = (ndpi+1)+i*ndpi # nodo incial cada nivel
    for j in range(ndpi):
        if i != nPz-1:
            ops.mass(nodoi+j, *[mnd[j], mnd[j], 0.0]) # asignando las masa
        else:
            ops.mass(nodoi+j, *[mndt[j], mndt[j], 0.0])

# Eigevalores y Eigenvectores

vals = np.array(ops.eigen('-fullGenLapack', Nmodes)) # eigenvalores
omega = np.sqrt(vals) # frecuencia angular (rad/s)
Tmodes = 2*np.pi/omega # periodo (s)
frec = 1/Tmodes # frecuencia (Hz o s^-1)

# tabular resultados
df = pd.DataFrame(columns = ['Modo', 'w (rad/s)', 'f (Hz)', 'T (s)'])
for i in range(Nmodes):
    df = df._append({'Modo': i+1, 'w (rad/s)': omega[i], 'f (Hz)': frec[i], 'T (s)': Tmodes[i]}, ignore_index = True)

df = df.astype({'Modo': int})
display(df.round(2))

# Graficar modos de vibración
fmt_undefo = {'color': 'gray', 'linestyle': (0,(1.2,2.0)), 'linewidth': 2.0,
              'marker': '', 'markersize': 2.0}
for i in range(Nmode):
    opsv.plot_mode_shape(i+1,endDispFlag = 0, fig_wi_he = (30,35), 
                         fmt_undefo = fmt_undefo, 
                         node_supports = False,az_el = (-140,30))
    plt.title(f'T[{i+1}]: {Tmodes[i]: .2f} s')
    plt.savefig(f'ModShape{i+1}.png', dpi = 300, bbox_inches = 'tight')

# Análsisis Modal Espectral

# Obtener la matriz de masas
ops.wipeAnalysis()
ops.system('FullGeneral')
ops.numberer('Plain')
ops.constraints('Transformation')
ops.algorithm('Linear')
ops.analysis('Transient')
ops.integrator('GimmeMCK', 1.0, 0.0, 0.0)
ops.analyze(1, 0)

# Matriz de masa (Ms)
NGDL = ops.systemSize() # GDL = 3gdl*Npisos*(ndpi+1CM)
Mmatriz = ops.printA('-ret')
Mmatriz = np.array(Mmatriz)
Mmatriz.shape = (NGDL, NGDL)
Ms = Mmatriz[-3*nPz:, -3*nPz:]

# Obtener los modos de vibración
Tags = ops.getNodeTags() # para obtener etiqueta de nodos = ndpi*(Npisos+1)+NCM
# print(Tags)

# Formas Modales (3GDL/piso)
modo = np.zeros((Nmodes, 3*nPz))
for j in range(1, Nmodes+1):
    ind = 0
    for i in Tags[-nPz:]:
        temp = ops.nodeEigenvector(i, j)
        modo[j-1, [ind, ind+1, ind+2]] = temp[0], temp[1], temp[-1]
        ind = ind+3

# Definir valores iniciales
ux, uy, rz = np.zeros(3*nPz), np.zeros(3*nPz), np.zeros(3*nPz)
ux[0::3], uy[1::3], rz[2::3] = 1.0, 1.0, 1.0
sumux, sumuy, sumrz = 0.0, 0.0, 0.0
ni = 0
Mux = sum(sum(Ms[0::3, 0::3])) # masa traslacional en x (kg)
Muy = sum(sum(Ms[1::3, 1::3])) # masa traslacional en y (kg)
Mrz = sum(sum(Ms[2::3, 2::3])) # masa rotacional en z (kg.m^2)

# Tabular resultados: contribución de masa de los modos
df1 = pd.DataFrame(columns = ['Modo', 'T (s)', 'sum_ux', 'sum_uy', 'sum_rz'])
for j in range(1,Nmodes+1):
    FPux = modo[j-1].T@ Ms @ ux
#             1x15     15x15    15x1
    FPuy = modo[j-1].T@ Ms @ uy
    FPrz = modo[j-1].T@ Ms @ rz
    FPRux, FPRuy, FPRrz = FPux**2/Mux, FPuy**2/Muy, FPrz**2/Mrz
    sumux, sumuy, sumrz = sumux+FPRux, sumuy+FPRuy, sumrz+FPRrz
    if min(sumux, sumuy, sumrz) >= 0.90 and ni == 0:
        ni = j
    df1 = df1._append({'Modo': j, 'T (s)': Tmodes[j-1], 'sum_ux': sumux, 'sum_uy': sumuy, 'sum_rz': sumrz}, ignore_index = True)

df1 = df1.astype({'Modo': int})
display(df1.round(3))
print(f'Cantidad de modos requeridos: {ni}')
nmreq = ni

# Análisis Modal Espectral: Superposión Modal
def AnalisisModal(Aa, Av, I, TS, Ms, modo, Tmodes, NGDL, ni, ux, uy, rz, FAx, FAy):
    g = 9.81
    # Valores iniciales:
    D_RCSCx, Δ_RCSCx, V_RCSCx = np.zeros(NGDL), np.zeros(NGDL), np.zeros(NGDL)
    D_RCSCy, Δ_RCSCy, V_RCSCy = np.zeros(NGDL), np.zeros(NGDL), np.zeros(NGDL)
    
    for j in range(1, ni+1):
        FPux = modo[j-1].T@ Ms @ ux
        FPuy = modo[j-1].T@ Ms @ uy
        
        Sa = SaNSR10(Tmodes[j-1], Aa, Av, I, TS)*g
        Sd = (Tmodes[j-1]/(2*np.pi))**2 * Sa
        
        # Respuesta en x
        respDx = Sd*FPux*modo[j-1]*FAx
        respAx = Sa*FPux*Ms @ modo[j-1]*FAx
        D_RCSCx = D_RCSCx + (respDx)**2
        respDx[3:] = respDx[3:] - respDx[:-3]
        Δ_RCSCx = Δ_RCSCx + respDx**2
        V_RCSCx = V_RCSCx + (np.cumsum(respAx[::-1])[::-1])**2
        
        # Respuesta en y
        respDy = Sd*FPuy*modo[j-1]*FAy
        respAy = Sa*FPuy*Ms @ modo[j-1]*FAy
        D_RCSCy = D_RCSCy + (respDy)**2
        respDy[3:] = respDy[3:] - respDy[:-3]
        Δ_RCSCy = Δ_RCSCy + respDy**2
        V_RCSCy = V_RCSCy + (np.cumsum(respAy[::-1])[::-1])**2
    
    # Obtener la respuesta
    Dx, Δx, Vx = D_RCSCx**0.5, Δ_RCSCx**0.5, V_RCSCx**0.5
    Dy, Δy, Vy = D_RCSCy**0.5, Δ_RCSCy**0.5, V_RCSCy**0.5
    
    # Tabular resultados: V (kN), D (cm)
    df = pd.DataFrame(columns = ['Nivel', 'Vx (kN)', 'Vy (kN)', 'Dx (cm)', 'Dy (cm)'])
    for i in range(int(NGDL/3)):
        df = df._append({'Nivel': i+1, 'Vx (kN)': Vx[0::3][i]*1e-3, 'Vy (kN)': Vy[1::3][i]*1e-3,
                          'Dx (cm)': Dx[0::3][i]*1e2, 'Dy (cm)': Dy[1::3][i]*1e2}, ignore_index = True)
    return Dx, Δx, Vx, Dy, Δy, Vy, df


# Espectro elástico de Diseño (NSR-10)
def SaNSR10(T, Aa, Av, I, TS):
    # Definición de los tipos de suelo (Letra: Número)
    TS_tx2num = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    
    # Convertir el tipo de suelo TS de letra a número
    Ts_num = TS_tx2num.get(TS)
    
    if Ts_num is None:
        raise ValueError('Tipo de suelo no encontrado.')
    
    # Definición de los factores Fa y Fv para cada tipo de suelo
    factores_de_suelo = {
        1: (0.8, 0.8),
        2: (1, 1),
        3: (
            1.2 if Aa <= 0.2 else (1.0 - 1.2) / (0.4 - 0.2) * (Aa - 0.2) + 1.2 if Aa <= 0.1 else 1,
            1.7 if Av <= 0.1 else (1.3 - 1.7) / (0.5 - 0.1) * (Av - 0.1) + 1.7 if Av <= 0.5 else 1.3
            ),
        4: (
            1.6 if Aa <= 0.1 else (
                (1.2 - 1.6) / (0.3 - 0.1) * (Aa - 0.1) + 1.6 if Aa <= 0.3 else (1.0 - 1.2) / (0.5 - 0.3) * (Aa - 0.3) + 1.2 if Av <= 0.5 else 1.0
                                  ),
            2.4 if Av <= 0.1 else (
                (2 - 2.4) / (0.2 - 0.1) * (Av - 0.1) + 2.4 if Av <= 0.2 else (
                    (1.6 - 2) / (0.4 - 0.2) * (Av - 0.2) + 2 if Av <= 0.4 else (1.5 - 1.6) / (0.5 - 0.4) * (Av - 0.4) + 1.6 if Av <= 0.5 else 1.5
                                                                              )
                                  ),
            ),
        5: (
            2.5 if Aa <= 0.1 else (
                (1.7 - 2.5) / (0.2 - 0.1) * (Aa - 0.1) + 2.5 if Aa <= 0.2 else (
                    (1.2 - 1.7) / (0.3 - 0.2) * (Aa - 0.2) + 1.7 if Aa <= 0.3 else (0.9 - 1.2) / (0.4 - 0.3) * (Aa - 0.3) + 1.2 if Aa <= 0.4 else 0.9
                                                                                )
                                  ),
            3.5 if Av <= 0.1 else (
                (3.2 - 3.5) / (0.2 - 0.1) * (Av - 0.1) + 3.5 if Av <= 0.2 else (
                    (2.4 - 3.2) / (0.4 - 0.2) * (Av - 0.2) + 3.2 if Av <= 0.4 else 2.4
                                                                                )
                                  ),
            )
                        }

    # Obtener los factores correspondientes al tipo de suelo
    Fa, Fv = factores_de_suelo[Ts_num]
    
    # Calcular los periodos
    T0 = 0.1 * (Av * Fv) / (Aa * Fa)
    TC = 0.48 * (Av * Fv) / (Aa * Fa)
    TL = 2.4 * Fv

    # Determinar la respuesta espectral según el periodo
    if T <= T0:
        return 2.5 * Aa * Fa * I
    elif T <= TC:
        return 2.5 * Aa * Fa * I
    elif T <= TL:
        return 1.2 * Av * Fv * I / T
    else:
        return 1.2 * Av * Fv * TL * I / (T ** 2)

# Graficar el espectro de diseño
Tnf = 5.0 # periodo final (s) para graficar
dTn = 0.01 # paso del Periodo (s)
Tn = np.arange(0.0, Tnf+0.01, dTn)

San = [SaNSR10(T, Aa, Av, I, TS) for T in Tn]

plt.figure(figsize = (7, 4))
plt.plot(Tn, San, color = 'b', linewidth = 1.5)
plt.title("Espectro de diseño", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$T$ (s)',fontsize=14)
plt.ylabel(r'$S_a$ (g)',fontsize=14)
plt.axis([Tn[0], Tn[-1], 0.0, 0.8])
plt.grid(False)
plt.savefig('Espectro_NSR10.png', dpi = 300, bbox_inches = 'tight')
plt.show()

NGDL = 3*nPz

ni = Nmodes if ni == 0 else ni

Dx, Δx, Vx, Dy, Δy, Vy, df2 = AnalisisModal(Aa, Av, I, TS, Ms, modo, Tmodes, NGDL, ni, ux, uy, rz, FAx, FAy)

print('Análsis Modal Espectral')
df2 = df2.astype({'Nivel': int})
display(df2.round(2))

df3 = pd.DataFrame(columns = ['Nivel', 'Vx (kN)', 'Vy (kN)', 'Dx (cm)', 'Dy (cm)', 'Δx (%)', 'Δy (%)'])
# Si se incluyen secciones fisuradas, las derivas se multiplcan por 0.7
RDER = 1.00 if IncSF == 0 else 0.70

for i in range(nPz):
    rΔx = Δx[0::3][i]/dHz*RDER # relación de deriva de entrepiso en x
    rΔy = Δy[1::3][i]/dHz*RDER # relación de deriva de entrepiso en y
    df3 = df3._append({'Nivel': i+1, 'Vx (kN)': Vx[0::3][i]*1e-3, 'Vy (kN)': Vy[1::3][i]*1e-3,
                      'Dx (cm)': Dx[0::3][i]*1e2, 'Dy (cm)': Dy[1::3][i]*1e2,
                      'Δx (%)': rΔx*1e2, 'Δy (%)': rΔy*1e2}, ignore_index = True)
df3 = df3.astype({'Nivel': int})
display(df3.round(2))

# Graficar la relación de deriva
vecx = np.array(df3.loc[:,'Δx (%)'])
vecy = np.array(df3.loc[:,'Δy (%)'])
vecVx = np.array(df3.loc[:,'Vx (kN)'])
vecVsx = np.append(np.repeat(vecVx, 2), 0)
Vstx = vecVx.max()
vecVy = np.array(df3.loc[:,'Vy (kN)'])
vecVsy = np.append(np.repeat(vecVy, 2), 0)
Vsty = vecVy.max()
Dermax = vecx.max()
Dermay = vecy.max()
vpis = np.repeat(np.arange(nPz+1), 2)[1:]

lim2 = 1.1*max(vecVx.max(), vecVy.max())

lim = 1.1*max(vecx.max(), vecy.max())

plt.figure(figsize = (4,8))
plt.plot(np.insert(vecx,0,0), np.arange(nPz+1), 'bo-', label = 'en x', lw = 0.8)
plt.plot(np.insert(vecy,0,0), np.arange(nPz+1), 'ro-', label = 'en y', lw = 0.8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.xlabel('Relación de deriva (%)',fontsize=14)
plt.ylabel('Nivel',fontsize=14)
plt.axis([-0.1, lim+0.1, -0.1, nPz+0.1])
plt.grid(False)
plt.savefig('Relacion_de_Deriva.png', dpi = 300, bbox_inches = 'tight')
plt.show()

plt.figure(figsize = (4.0, 8.0))
plt.plot(vecVsx, vpis, 'b-', label = 'en x', lw = 1.6, ms = 6)
plt.plot(vecVsy, vpis, 'r-', label = 'en y', lw = 1.6, ms = 6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.xlabel('Cortante (kN)',fontsize=14)
plt.ylabel('Nivel',fontsize=14)
plt.axis([-0.1, lim2+0.1, -0.1, nPz+0.1])
plt.grid(False)
plt.savefig('Cortante_Basal.png', dpi = 300, bbox_inches = 'tight')
plt.show()

# Cálculo del corte mínimo
Ct, alpha = 0.047, 0.9 # Tabla A.4.2-1
Hedf = nPz*dHz # altura de la edificación (m)
Ta = Ct*Hedf**alpha # Periodo aproximado (s)
Saa = SaNSR10(Ta, Aa, Av, I, TS)
Vs = Mux*Saa*g*1e-3
FaVsx = 0.8*Vs/Vstx
FaVsy = 0.8*Vs/Vsty

print('---------------------------------------')
print('Fuerza Horizontal Equivalente')
print('---------------------------------------')
print(f'T_a = {round(Ta, 2)} s')
print(f'V_s = {round(Vs, 2)} kN')
print('---------------------------------------')
print('Superposición modal')
print('---------------------------------------')
print(f'Cantidad de modos requeridos: {nmreq}')
print(f'Deriva_máxima = {round(Dermax, 2)} %')
print(f'V_stx = {round(Vstx, 2)} kN')
print('Mantener FAx = 1.00' if FaVsx <= 1.0 else f'Modificar FAx = {round(FaVsx, 2)}')
print(f'V_sty = {round(Vsty, 2)} kN')
print('Mantener FAy = 1.00' if FaVsy <= 1.0 else f'Modificar FAy = {round(FaVsy,2)}')
