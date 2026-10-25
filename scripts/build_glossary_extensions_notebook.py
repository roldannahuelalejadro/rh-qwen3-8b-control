"""Execute and document the four requested decoder checks."""
import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/glossary/decoder_v2_checks'
OUT.mkdir(exist_ok=True)
PATH=ROOT/'notebooks/glossary/04_ajuste_trayectorias_constante_cubico.ipynb'
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=textwrap.dedent(s).strip()))
def code(s):cells.append(dict(cell_type='code',metadata={},source=textwrap.dedent(s).strip(),execution_count=None,outputs=[]))

md(r'''
# Ajuste, todos los j, sensibilidad al intercepto y orden cubico

Seguimiento solicitado sobre la primera prueba. Se conservan las etiquetas
manuales, los 72 rollouts clasificables, las mismas particiones externas,
los mismos cinco pasos de entrenamiento (j=0,32,64,128,192), K=1 y las 36 capas.
Los 28 X no participan del ajuste, PCA ni seleccion de hiperparametros.
La adquisicion original y los resultados v1 se conservan.
''')

md(r'''
## 1. Como se ajusto el decoder original

Se entreno una regresion logistica L2, no cuadrados minimos. Los cuadrados
minimos eran el procedimiento del trabajo anterior con una dificultad continua.
Aqui el objetivo observado es una etiqueta manual binaria por rollout.

Para cada capa, PCA se ajusto sobre los promedios acumulativos de los rollouts
de entrenamiento y sus cinco pasos j. Se retuvo una componente por capa. Sus
coordenadas se estandarizaron con media y desviacion del mismo entrenamiento.
Llamemos z al vector de estas 36 coordenadas estandarizadas y phi(z) a sus
monomios de grado uno y dos, sin repetir los productos simetricos.

$$s=b_z+w^\top\phi(z),\qquad p=\frac{1}{1+\exp(-s)}.$$

La implementacion original utilizo `LogisticRegression` de scikit-learn con
solver `liblinear`, `penalty='l2'`, maximo 3000 iteraciones, tolerancia por defecto
1e-4 y `intercept_scaling=100`. En su convencion de suma de perdidas, el objetivo es

$$\mathcal L=\sum_n[\log(1+\exp(s_n))-Y_n s_n]
+\frac{\lVert w\rVert_2^2}{2C}+\frac{b_z^2}{2C\,100^2}.$$

El intercepto tiene una penalizacion pequena, no exactamente cero. w contiene
coeficientes de monomios; la norma usada no es directamente la norma de
Frobenius de H. Para escribir 1/2 x^T H x, la diagonal de H lleva dos veces
el coeficiente del monomio cuadrado y los elementos cruzados llevan el
coeficiente del monomio correspondiente. Se guardo ademas la transformacion
de los coeficientes a las coordenadas PCA sin estandarizar.

Tres folds externos separaron rollouts completos. Dentro de cada entrenamiento
externo, otros tres folds eligieron C en {0.001,0.01,0.1,1}, minimizando log-loss
de validacion sobre los cinco pasos. PCA y escalado se reajustaron dentro de
cada entrenamiento interno. Todos los prefijos de un rollout quedaron juntos.
La etiqueta final se repite para los cinco estados, que no son cinco muestras
independientes. Los mismos parametros del decoder se utilizan para todo j.
''')

md(r'''
## 2. Trayectorias token a token

Ahora se calcula una prediccion para **cada j entero de 0 a 192**, para los mismos
72 rollouts y con modelos externos que no los vieron durante el ajuste.
Para r0001 y r0026 se extiende la curva hasta el ultimo token valido,
incluido EOS en r0026. No se usa padding.

El encoder es lineal y fijo: podemos proyectar los tokens y acumular esas
proyecciones para obtener el mismo promedio proyectado. Las diferencias de
redondeo frente a la cache float32 v1 se registran por rollout.

El entrenamiento sigue usando los cinco pasos declarados; evaluar cada token
no significa reajustar cada token. Despues de j=192 las curvas de los dos casos
son extrapolaciones del horizonte temporal y se muestran en zona gris y con
linea discontinua. Las metricas poblacionales se limitan a 0..192 para no cambiar
la cohorte segun que respuestas ya terminaron.
''')

md(r'''
## 3. Que significa la constante y que prueba hacemos

El intercepto es un score sin unidades de probabilidad. Su valor puede ser
negativo o mayor que uno. Ademas depende del origen de coordenadas: trasladar
las coordenadas cambia b y g sin cambiar las predicciones, si se transforman
coherentemente los coeficientes. En el modelo original, el estado inicial no
esta necesariamente en x=0, por lo que b no es por si solo el score inicial.

Para hacer interpretable la prueba, centramos las caracteristicas del
decoder en el prompt compartido, sin modificar las direcciones PCA:

$$F(z)=\phi(z)-\phi(z_0),\qquad s_j=\beta+w^\top F(z_j),\qquad s_0=\beta.$$

Se conservan las mismas penalizaciones de los coeficientes w. Aqui beta es
exactamente el logit inicial del prompt. Se prueban los cinco valores
0,0.25,0.5,0.75,1, que corresponden a probabilidades iniciales entre 0.5 y
0.731 aproximadamente; NO a probabilidades entre cero y uno.
Las probabilidades exactamente 0 y 1 requeririan logits infinitos.

Se distinguen dos experimentos:

- **Inicializacion:** beta empieza en cada valor y despues se optimiza junto
  con w. Se comparan iteraciones, objetivo, gradiente y predicciones finales.
- **Constante fija:** beta permanece en ese valor y solo se optimiza w.
  Esto cambia el problema de ajuste; la convergencia no puede eliminar una
  restriccion fija.

Se usa L-BFGS-B con gradiente analitico, maxiter=2000, gtol=1e-8 y ftol=1e-13:

$$\mathcal L=\frac1n\sum_n[\log(1+\exp(s_n))-Y_n s_n]
+\frac{\lVert w\rVert_2^2}{2Cn}.$$

En esta prueba beta no se penaliza. C se mantiene en el valor elegido en los
folds internos del cuadratico original, sin consultar el test. Cada beta se
ajusta con los mismos datos y el mismo C dentro de cada fold. Para encoder fijo,
la regresion es convexa en sus coeficientes; cambiar la inicializacion no deberia
producir soluciones distintas al converger. Se verifica numericamente.

El baseline constante original predice la prevalencia de entrenamiento:
51/72=0.7083 globalmente. Su logit es log(51/21), aproximadamente 0.8873.
Esto describe este prompt y este criterio manual, no una tendencia universal
de la LLM. No tenemos respuestas de un prompt como "banana split" en esta
adquisicion: esta prueba no establece generalizacion entre prompts. Modificar
el intercepto no reemplaza medirlos e incorporarlos al entrenamiento y evaluacion.
''')

md(r'''
## 4. Un orden mas: termino cubico

Se interpreta la propuesta como extender la expansion hasta grado tres:

$$s_j^{(b)}=b_0+\sum_a g_a x_{aj}^{(b)}
+\frac12\sum_{a,c}H_{ac}x_{aj}^{(b)}x_{cj}^{(b)}
+\frac1{3!}\sum_{a,c,d}U_{acd}x_{aj}^{(b)}x_{cj}^{(b)}x_{dj}^{(b)}.$$

Los indices a,c,d agrupan direccion PCA y capa; b identifica el rollout.
El tensor U es simetrico y permite dependencias de tercer orden. Con d=36
coordenadas, se agregan 8436 monomios y se pasa de 703 a **9139 parametros**,
frente a solo 72 rollouts. La regularizacion se elige con el mismo esquema
anidado y la misma grilla, sin elegir el grado con el test. Es una comparacion
exploratoria posterior al resultado cuadratico, no una confirmacion independiente.

Se guardan coeficientes y tensor U en coordenadas estandarizadas. Que un modelo
de mayor orden mejore el ajuste de entrenamiento no basta para justificarlo:
la comparacion relevante es la log-loss y discriminacion fuera de entrenamiento.
''')

code(r'''
from pathlib import Path
import sys
import json
RAIZ=next((p for p in (Path.cwd(),*Path.cwd().parents)
           if (p/'scripts/glossary_decoder_extensions.py').exists()),None)
if RAIZ is None:
    raise FileNotFoundError('Abra el notebook en el workspace o indique RAIZ.')
sys.path.insert(0,str(RAIZ/'scripts'))
from glossary_decoder_extensions import run, figures
SALIDA=RAIZ/'analysis/glossary/decoder_v2_checks'
resultado=run(RAIZ,SALIDA)
print(json.dumps(resultado['summary'],indent=2))
''')
code(r'''
rutas_figuras=figures(resultado,SALIDA)
print('\n'.join(str(p) for p in rutas_figuras))
''')

def save():
    PATH.write_text(json.dumps(dict(nbformat=4,nbformat_minor=4,cells=cells,metadata={
        'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python','file_extension':'.py'}}),indent=1,ensure_ascii=False),encoding='utf-8')
class Tee:
    def __init__(self,b,s):self.b,self.s=b,s
    def write(self,v):self.b.write(v);return self.s.write(v)
    def flush(self):self.s.flush()
namespace={'__name__':'__main__'}
save()
count=0
for cell in cells:
    if cell['cell_type']!='code':continue
    count+=1
    buffer=io.StringIO()
    with contextlib.redirect_stdout(Tee(buffer,sys.stdout)):
        exec(compile(cell['source'],f'extension_cell_{count}','exec'),namespace)
    cell['execution_count']=count
    cell['outputs']=[dict(output_type='stream',name='stdout',text=buffer.getvalue())]
    save()
for p in namespace['rutas_figuras']:
    count+=1
    source=f"from IPython.display import Image, display\ndisplay(Image(filename=str(SALIDA/'{p.name}')))"
    compile(source,'image_cell','exec')
    cells.append(dict(cell_type='code',metadata={},source=source,execution_count=count,
        outputs=[dict(output_type='display_data',metadata={},data={
            'image/png':base64.b64encode(p.read_bytes()).decode(),'text/plain':p.name})]))
save()
print('Notebook saved:',PATH,flush=True)
