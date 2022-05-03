import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly_express as px
import copy
from scipy import special
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits import mplot3d






primaryColor="#d33682"
backgroundColor="#002b36"
secondaryBackgroundColor="#586e75"
textColor="#fafafa"
font="sans serif"

link = '[Daniil Davidov](https://vk.com/daniil2510_davidov)' 

st.markdown("<center><h1>Численное и аналитическое решения уравнения теплопроводности.</h1>\
   <h4> Давыдов Даниил,  <br>  \
    физический факультет МГУ,  <br>  \
    кафедра физики частиц и космологии,  <br>  \
    343 группа    </h4></center>",  unsafe_allow_html=True)
st.markdown(  link, unsafe_allow_html=True)
st.sidebar.markdown("<h1><a href=\"#top\">Содержание </a></h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2><a href=\"#top1\">1. Введение. Постановка задачи </a></h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2><a href=\"#top2\">2. Аналитическое решение </a></h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2><a href=\"#top3\">3. Численное решение  </a></h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2><a href=\"#top4\">4. Графики </a></h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2><a href=\"#top5\">5. Итоги работы </a></h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2><a href=\"#top6\">6. Приложение. Код </a></h2>", unsafe_allow_html=True)


st.markdown("<h1><a name=\"top\">Содержание</a></h1>", unsafe_allow_html=True)




st.markdown("<h2><a name=\"top1\">1. Введение. Постановка задачи </a></h2>", unsafe_allow_html=True)
st.markdown('<h3>1.1. Модель</h3>',unsafe_allow_html=True)
st.markdown('<p style="text-indent: 25px;"> В настоящее время есть разные методы изучения и познания окружающего нас мира.\
 Различают два уровня научного познания: эмпирический и теоретический. Одни общенаучные методы \
 применяются только на эмпирическом уровне (наблюдение, эксперимент), другие — только на \
 теоретическом (идеализация, формализация). Общенаучный метод, который используется как на \
 эмпирическом, так и на теоретическом уровне, является моделирование. Соответственно можно дать\
  такое определение:\
<p style="text-indent: 25px;">Моделирование — метод познания окружающего мира, относящийся\
 к общенаучным методам, применяемым как на эмпирическом, так и на теоретическом уровне познания.</p>\
<p style="text-indent: 25px;">Под моделью понимается такой материальный или мысленно представляемый\
  объект, который в процессе познания (изучения) замещает объект-оригинал,\
 сохраняя  некоторые важные для данного исследования типичные его черты.\
  Сам процесс построения и использования модели и называется моделированием.<p>\
<p style="text-indent: 25px;">Определние модели можно дать другими словами:<p>\
<p style="text-indent: 25px;">Модель — это объект-заменитель объекта-оригинала,\
 обеспечивающий изучение некоторых интересующих исследователя свойств оригинала.<p>\
<p style="text-indent: 25px;">Заметим, что любая модель является неполной (нетождественна\
 объекту-оригиналу), так как при ее построении исследователь учитывал лишь важнейшие с его\
  точки зрения факторы. Поэтому, "полная" модель будет полностью тождественна оригиналу.<p>\
<p style="text-indent: 25px;">Говорят, что модель адекватна объекту, если результаты\
 моделирования удовлетворяют исследователя и могут служить основой для прогнозирования \
 поведения или свойств исследуемого объекта. Адекватность модели зависит от целей моделирования\
 и принятых критериев.<p>\
<p style="text-indent: 25px;"><p>\
<p style="text-indent: 25px;"><p>\
  ', unsafe_allow_html=True)


st.markdown('<h3>1.2. Математическое моделирование</h3>',unsafe_allow_html=True)

st.markdown('<p style="text-indent: 25px;">Математическое моделирование — это идеальное\
 научное знаковое формальное моделирование, при котором описание объекта осуществляется на языке\
 математики, а исследование модели проводится с использованием тех или иных математических\
  методов.<p>\
<p style="text-indent: 25px;">В настоящее время математическое моделирование является одним\
 из самых результативных и наиболее применяемых методов научного исследования. Все современные\
  разделы физики посвящены построению и исследованию математических моделей различных физических\
  объектов и явлений. Так, например, физики - "ядерщики" до проведения экспериментов выполняют \
  серьезные исследования с применением математических моделей.<p>\
<p style="text-indent: 25px;">В данной работе будет рассмотрена краевая задача для уравнения\
 теплопроводности. С применением метода переменных направлений и  метода прогонки\
 будет получено численное решение, после чего оно будет сравниваться с аналитическим решением.\
 Будут обсуждаться вопросы об устойчивости и сходимости данной схемы, ошибки численного решения\
 в зависимости от сетки, и построены соответсвующие графики. <p>\
<p style="text-indent: 25px;">Соответствующая кравевая задача уравнения теплопроводности \
(параболического типа) выглядит следующим образом:\
<p>',unsafe_allow_html=True)

st.markdown('<h3>1.3. Постановка задачи</h3>',unsafe_allow_html=True)
st.markdown(' <p style="text-indent: 25px;">Соответствующая кравевая задача уравнения теплопроводности \
(параболического типа) выглядит следующим образом:\
<p>',unsafe_allow_html=True)

st.latex(r'''
     \begin{equation}
\left\{
\begin{aligned}
&\frac{\partial u}{\partial t} =  \Delta u + e^{-t}cosxcosy, \; x \in (0,\pi), \; y \in (0,\pi), \; t > 0, \\
&\frac{\partial u}{\partial x}\bigg|_{x=0} = \frac{\partial u}{\partial x}\bigg|_{x=\pi} = 0, \\
&\frac{\partial u}{\partial y}\bigg|_{y=0} = \frac{\partial u}{\partial y}\bigg|_{y=\pi} = 0, \\
&u|_{t=0} = cosxcosy \;
\end{aligned}
\right.
\end{equation}
     ''')

st.markdown("<h2><a name=\"top2\">2. Аналитическое решение </a></h2>", unsafe_allow_html=True)


st.markdown('<p style="text-indent: 25px;">Прежде всего нам стоит решить задачу аналитически. \
    Перепишем еще раз задачу:<p>\
  ', unsafe_allow_html=True)
st.latex(r'''
     \begin{equation}
\left\{
\begin{aligned}
&\frac{\partial u}{\partial t} =  \Delta u + e^{-t}cosxcosy, \; x \in (0,\pi), \; y \in (0,\pi), \; t > 0, \\
&\frac{\partial u}{\partial x}\bigg|_{x=0} = \frac{\partial u}{\partial x}\bigg|_{x=\pi} = 0, \\
&\frac{\partial u}{\partial y}\bigg|_{y=0} = \frac{\partial u}{\partial y}\bigg|_{y=\pi} = 0, \\
&u|_{t=0} = cosxcosy \;
\end{aligned}
\right.
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Найдем собственные значения и собственные функции \
    вспомогательной задачи Штурма-Лиувилля:', unsafe_allow_html=True)

st.latex(r'''
     \begin{equation}
\left\{
\begin{aligned}
&\Delta V + \lambda V = 0, \; x \in (0,\pi), \; y \in (0,\pi), \\
&\frac{\partial V}{\partial x}\bigg|_{x=0} = \frac{\partial u}{\partial x}\bigg|_{x=\pi} = 0, \\
&\frac{\partial V}{\partial y}\bigg|_{y=0} = \frac{\partial u}{\partial y}\bigg|_{y=\pi} = 0, \;
\end{aligned}
\right.
\end{equation}
     ''')
st.markdown('<p style="text-indent: 25px;">Функцию V будем искать в виде (метод разделения переменных) ', unsafe_allow_html=True)

st.latex(r'''
V(x,y) = X(x)Y(y) 
     ''')

st.markdown('<p style="text-indent: 25px;">Подставляя эту функцию в уравнение для V, получим\
 задачи на собственные значения: ', unsafe_allow_html=True)


st.latex(r'''
 \begin{equation}

\begin{aligned}
&X^{\prime\prime}Y + Y^{\prime\prime}X + \lambda XY = 0 \\
&\frac{ X^{\prime\prime}}{X}  = -  \frac{ Y^{\prime\prime}}{Y} - \lambda = \mu , \;
\end{aligned}

\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;"> Рассмотрим первую задачу:  ', unsafe_allow_html=True)

st.latex(r'''
 \begin{equation}
\left\{
\begin{aligned}
&X^{\prime\prime}(x) + \lambda X(x) = 0, x \in (0,\pi), \\
&X^{\prime}(0) = X^{\prime}(\pi) = 0, \; 
\end{aligned}
\right.
\end{equation}
     ''')


st.markdown('<p style="text-indent: 25px;"> Ее решение: <p>  ', unsafe_allow_html=True)

st.latex(r'''
\begin{equation}
X_{n}(x) =cos(nx), \; \mu_{n} =n^2,\; n=0, 1,2,\dots
\end{equation}

     ''')

st.markdown('<p style="text-indent: 25px;"> Аналогично для второй функции получим: <p>  ', unsafe_allow_html=True)


st.latex(r'''
\begin{equation}
Y_{n}(x) =cos(my), \; \nu_{m} =m^2,\; m=0, 1,2,\dots
\end{equation}

     ''')

st.markdown('<p style="text-indent: 25px;">Тогда функция V(x,y) будет равна: <p>  ', unsafe_allow_html=True)

st.latex(r'''
\begin{equation}
V_{nm}(x,y) =cos(nx)cos(my), \; \nu_{m} =m^2,\; n, m=0, 1,2,\dots
\end{equation}

     ''')

st.markdown('<p style="text-indent: 25px;"> А соответсвующее собственное значение: <p>  ', unsafe_allow_html=True)


st.latex(r'''
\begin{equation}
\lambda_{nm}  = n^2 + m^2 \;
\end{equation}

     ''')

st.markdown('<p style="text-indent: 25px;"> Неоднородность в уравнении разложим по \
ортонормированной системе {Vnm(x,y)}<p>  ', unsafe_allow_html=True) 



st.latex(r'''
 \begin{equation}

\begin{aligned}
&f(x,y,t) = e^{-t}cosxcosy = e^{-t}V_{11}(x,y) = \sum_{n=0}^\infty \sum_{m=0}^\infty f_{nm}(t)V_{nm}(x,y), \\
&f{nm}(t) = e^{-t}  \delta_{n1}  \delta_{m1} \; 
\end{aligned}
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;"> Тогда решение будет записываться в виде\
    (граничные условия будут выполнены \
    автоматически):<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
u(x,y,t) = \sum_{n=0}^\infty \sum_{m=0}^\infty T_{nm}(t)V_{nm}(x,y), \\ 
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Дальше подставляем это решение в дифференциальное\
    уравнение и начальные условия (2) \
    (получим задачу Коши для временной части):<p>  ', unsafe_allow_html=True) 


st.latex(r'''
 \begin{equation}
\left\{
\begin{aligned}
&T^{\prime}_{nm}(t) + \lambda_{nm} T_{nm}(t) = e^{-t}  \delta_{n1}  \delta_{m1}, \\
&T_{nm}(0) =\delta_{n1}  \delta_{m1} \; 
\end{aligned}
\right.
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Рассмотрим случай n = 0, m = 1:<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\left\{
\begin{aligned}
&T^{\prime}_{01}(t) +  T_{01}(t) = 0, \\
&T_{01}(0) = 0  \; 
\end{aligned}
\right.
\end{equation}
     ''')

st.latex(r'''
 \begin{equation}
 \begin{aligned}
&T_{01} = Ce^{-t}, \\
&C = 0   \Rightarrow T_{01} \equiv 0 \;
\end{aligned}
\end{equation}
     ''' )

st.markdown('<p style="text-indent: 25px;">Аналогично<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
 \begin{aligned}
&T_{10} \equiv 0 \;
\end{aligned}
\end{equation}
     ''' )

st.markdown('<p style="text-indent: 25px;">Рассмотрим n = 1, m = 1. В этом случае мы получим\
    ненулевое решение <p>', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\left\{
\begin{aligned}
&T^{\prime}_{11}(t) +  2T_{11}(t) = e^{-t}, \\
&T_{11}(0) = 1  \; 
\end{aligned}
\right.
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Решение будем находить в виде суммы\
    общего решения однородного уравнения и частного решения неоднороднородного уравнения:<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\begin{aligned}
T_{11}(t) = Be^{-2t} + \widetilde{T}_{11}(t), \;
\end{aligned}
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Частное решение неоднороднородного уравнения \
    будем искать в виде:<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\begin{aligned}
\widetilde{T}_{11}(t) = Сe^{-t} + D, \;
\end{aligned}
\end{equation}
     ''')
st.markdown('<p style="text-indent: 25px;">Подставялем его в дифференциальное уравнение (16):<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\begin{aligned}
-Ce^{-t} + 2Ce^{-t} + 2D = e^{-t} \;
\end{aligned}
\end{equation}
     ''')
st.markdown('<p style="text-indent: 25px;">Получим:<p>  ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\begin{aligned}
&С = 1, \\
&D = 0  \\
&\widetilde{T}_{11}(t) = e^{-t}
\end{aligned}
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Откуда находим:\
    <p> \
 ', unsafe_allow_html=True) 


st.latex(r'''
 \begin{equation}
\begin{aligned}
T_{11}(t) = Be^{-2t} +e^{-t}
\end{aligned}
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Из начального условия:\
    <p> \
 ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\begin{aligned}
T_{11}(0) = B + 1 = 1, \\
B = 0, \\
\end{aligned}
\end{equation}
     ''')
st.markdown('<p style="text-indent: 25px;">Окончательно получаем решение:\
    <p> \
 ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
\begin{aligned}
T_{11}(t) = e^{-t}
\end{aligned}
\end{equation}
     ''')

st.markdown('<p style="text-indent: 25px;">Все слагаемые при n ≠ 1, m  ≠ 1 будут нулевыми.\
    Поэтому аналитическое решение задачи (2) будет следующим:\
    <p> \
 ', unsafe_allow_html=True) 

st.latex(r'''
 \begin{equation}
 \begin{aligned}
u(x,y,t) = \sum_{n=0}^\infty \sum_{m=0}^\infty T_{nm}(t)X_{n}(x)Y_{m}(y) = T_{11}(t)X_{1}(x)Y_{1}(y) = e^{-t}cosxcosy
\end{aligned}
\end{equation}
     ''')


st.markdown("<h2><a name=\"top3\">3. Численное решение </a></h2>", unsafe_allow_html=True)

st.markdown('<p style="text-indent: 25px;">В этой главе мы рассмотрим один из методов численного\
    решения уравнения теплопроводности. Первым шагом будет введение сетки (с учетом того, где заданы х и у):<p>  ', unsafe_allow_html=True) 

st.latex(r'''

\begin{aligned}
&w_{h} = \{x_{n}, y_{m}: x_{n} = nh_{x}, y_{m}, y_{m} = mh_{y}, n = \overline{0,N}, m = \overline{0,M},\
h_{x} = \frac{\pi}{N}, h_{y} = \frac{\pi}{M} \}, \\
&w_{\tau} = \{ t_{k} = k\tau \}, \\
&w_{h\tau} = w_{h} \otimes   w_{\tau} 
\end{aligned}
    ''')

st.markdown('<p style="text-indent: 25px;"> Дальше введем два оператора (разностная аппроксимация оператора Лапласа): <p>  ', unsafe_allow_html=True) 

st.latex(r'''

\begin{aligned}
\Lambda_{1}u = \frac{u_{n-1,m} - 2u_{nm} + u_{n+1,m}}{h_{x}^2}, \\
\Lambda_{2}u = \frac{u_{n,m-1} - 2u_{nm} + u_{n,m+1}}{h_{y}^2} \;
\end{aligned}
    ''')

st.markdown('<p style="text-indent: 25px;"> Следующим шагом будет использование\
 метода переменных направлений (обсуждение устойчивости и объема работы будет позже): <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}
\left\{
\begin{aligned}
&\frac{\overline{u}_{nm} - u_{nm}}{\tau / 2} = a^2(\Lambda_{1}\overline{u} + \Lambda_{2}u) + \overline{f} , \\
&\overline{u}^{\prime}|_{\gamma} = 0\;
\end{aligned}
\right.
\end{equation}
    ''')
    
st.latex(r'''
\begin{equation}
\left\{
\begin{aligned}
&\frac{\^u_{nm} - \overline{u}_{nm}}{\tau / 2} = a^2(\Lambda_{1}\overline{u} + \Lambda_{2}\^u) +\^f , \\
&\^u^{\prime}|_{\gamma} = 0  \\
\end{aligned}
\right.
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> где <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{aligned}
&\overline{u}_{nm} = u_{n,m, k+ 1/2} \\
&\^u_{nm} = u_{n,m, k+ 1} \;
\end{aligned}

    ''')

st.markdown('<p style="text-indent: 25px;"> Вначале решается уравнение (25), неявное по х и явное по у.\
Если аккуратно расписать операторы и учесть, что в граничном условии есть производная, получим систему из N+1\
 уравнений:  \
 <p>  ', unsafe_allow_html=True) 


st.latex(r'''
\begin{equation}
\left\{
\begin{aligned}
&\frac{a^2}{h_{x}^2} \overline{u}_{n+1,m} - \Big(\frac{2a^2}{h_{x}^2} + \frac{2}{\tau}\Big)\overline{u}_{n,m} + \frac{a^2}{h_{x}^2}\overline{u}_{n-1,m} = -F_{nm}, n =  \overline{1,N-1} \\
&\overline{u}_{0,m} = \overline{u}_{1,m}  \\
&\overline{u}_{N,m} = \overline{u}_{N-1,m} \;
\end{aligned}
\right.
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> где <p>  ', unsafe_allow_html=True) 


st.latex(r'''
\begin{aligned}
&F_{nm} = 2\Big( \frac{1}{\tau} - \frac{a^2}{h_{y}^2} \Big)u_{nm} + \frac{a^2}{h_{y}^2}\Big( u_{n,m-1} + u_{n,m+1} \Big) + \overline f, \\
&\overline f = e^{-\tau (k+1/2)}cosx_{n}cosy_{m}
\end{aligned}

    ''')

st.markdown('<p style="text-indent: 25px;"> И эта система уравнений решается для каждого m, где  <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{aligned}
m = \overline{1,M-1}
\end{aligned}

    ''')

st.markdown('<p style="text-indent: 25px;"> Для решения этой системы будем использовать\
 метод прогонки. Начнем с того, что будем искать решение в виде: <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}    
\begin{aligned}
\overline u_{n} = \alpha_{n+1} \overline u_{n+1} + \beta_{n+1} 
\end{aligned}
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> Согласно (28) предыдущее значение будет равно: <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}    
\begin{aligned}
\overline u_{n-1} = \alpha_{n} \overline u_{n} + \beta_{n} =  \alpha_{n} (\alpha_{n+1} \overline u_{n+1} + \beta_{n+1}) + \beta_{n}
\end{aligned}
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> Введем обозначения: <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}    
\begin{aligned}
&A_{n} = \frac{a^2}{h_{x}^2}, \\
&B_{n} = \frac{2a^2}{h_{x}^2} + \frac{2}{\tau}, \\
&C_{n} = \frac{a^2}{h_{x}^2}
\end{aligned}
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> Если подставить (29) в (27) и учесть (30), получим уравнение: <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}    
\begin{aligned}
&\overline u_{n+1} \Big( \alpha_{n+1}(C_{n}\alpha_{n} - B_{n}) + A_{n} \Big) + \Big( \beta_{n+1}(C_{n}\alpha_{n} - B_{n}) + C_{n}\beta_{n} + F_{n}  \Big) = 0\;
\end{aligned}
\end{equation}
    ''')
 
st.markdown('<p style="text-indent: 25px;"> Из этого уравнения находятся коэффициенты: <p>  ', unsafe_allow_html=True) 


st.latex(r'''
\begin{equation}    
\begin{aligned}
&\alpha_{n+1} = \frac{A_{n}}{B_{n} - C_{n}\alpha_{n}}  \\
&\beta_{n+1} = \frac{C_{n}\beta_{n} + F_{n}}{B_{n} - C_{n}\alpha_{n}} \;
\end{aligned}
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> Так как из (29) следует <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}    
\begin{aligned}
\overline{u_{0}} = \alpha_{1} \overline u_{1} + \beta_{1}
\end{aligned}
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;">то, пользуясь первым граничным условием из (27), найдем <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation}    
\begin{aligned}
&\alpha_{1} = 1, 
&\beta_{1} = 0 \;
\end{aligned}
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;">Все остальные коэффициенты можно найти, пользуясь рекурсивными формулами (32). \
Дальше из (28) получим: <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation} 
\left\{
\begin{aligned}
&\overline u_{N-1} = \alpha_{N} \overline u_{N} + \beta_{N}, \\
&\overline u_{N} = \alpha_{N} \overline u_{N-1} + \beta_{N} (\*)
\end{aligned}
\right.
\end{equation}
    ''')

st.markdown('<p style="text-indent: 25px;"> Применив второе граничное условие из (27), найдем <p>  ', unsafe_allow_html=True) 

st.latex(r'''
\begin{equation} 
\begin{aligned}
\overline u_{N} = \frac{\beta_{N}}{1-\alpha_{N}} \;
\end{aligned}
\end{equation}
    ''')


st.markdown('<p style="text-indent: 25px;"> Таким образом, для решения системы (27) требуется \
вначале найти все коэффициенты (при этом мы идем в прямом направлении от 1 до N), после чего по\
 формуле (36) находим вспомогательные решения (вспомогательные, так как эти решения\
  дальше будут использоваться для решения системы (26)), только теперь идя в обратном направлении от N до 1.\
   Все это делается для каждого m от 1 до M-1. Задача (26) решается аналогично, в результате\
    чего получим значения на новом слое к+1. При переходе от слоя к + 1 \
     (решение двух систем) к слою к + 2 процедура повторяется. <p>  ', unsafe_allow_html=True) 






st.markdown("<h2><a name=\"top4\">4. Графики </a></h2>", unsafe_allow_html=True)


st.markdown('<p style="text-indent: 25px;"> В этой главе мы строим графики для \
аналитического и численного решений и ошибки между ними.\
 При изменении параметров следует помнить, что ошибка вычисляется для одинаковых\
 моментов времени между численным и аналитическим решениями.  <p>  ', unsafe_allow_html=True) 



st.markdown('<h4>Здесь можно менять время и вращать график для аналитического решения:</h4>', unsafe_allow_html=True)
t = st.slider('время t', min_value= 0.0 , max_value=3.0, value = 0.1, step=0.1 )
r1 = st.slider('вращение по z', min_value= 0 , max_value=90, step=5, value = 20 )
r2 = st.slider('вращение в плоскости xy', min_value= -90 , max_value=90, step=5, value = -70 )

def f(x, y, t):
    return np.cos(x) * np.cos(y) * np.exp(-t)


x = np.linspace(0, np.pi, 101)
y = np.linspace(0, np.pi, 101)

X, Y = np.meshgrid(x, y)
Z = f(X, Y, t)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False,
cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Аналитическое решение')
ax.view_init(r1, r2 )
# fig.colorbar(surff, shrink=0.5, aspect=5)
ax.set_xticks([i for i in range(4)])
ax.set_yticks([i for i in range(4)])


ax.set_zticks([i - 1 for i in range(3)])
ax.zaxis.set_rotate_label(False)

figure = plt.gcf() #захват
st.pyplot(fig = figure)
plt.clf() #очищение 




st.markdown('<h3>Здесь можно менять время и вращать график для численного решения:</h3>', unsafe_allow_html=True)

tau = st.slider('шаг по времени ', min_value= 0.01 , max_value=0.1, value = 0.05,  step= 0.01  )

#tau = 0.05
k = 0
a = 0.; b = np.pi; c = np.pi; t = 3
u_left = 4.; u_right = -3.5
 # Определение числа интервалов сетки,
 # на которой будет искаться приближённое решение
N = 100
M = 100
hx = b/N
hy = c/M
T = 100
tk = k*tau
 # Определение cетки
x = np.linspace(a,b,N+1)
y = np.linspace(a,c,M+1)
t = np.linspace(a,t,T)



u = np.zeros((N+1,M+1,T))
for j in range(M+1):
    for i in range(N+1):
        u[i,j,0] = np.cos(x[i])*np.cos(y[j]) #начальные условия

        
An = 1/ (hx**2)
Bn = (2/(hx**2)) + 2/tau
Cn = 1/(hx**2)

alpha = np.zeros(N+1)
betha = np.zeros(N+1)
alpha[1] = 1   # здесь использованы граничные условия
alpha1 = alpha[1]  
betha1 = betha[1]  # здесь использованы граничные условия        



A1n = 1/ (hy**2)
B1n = (2/(hy**2)) + 2/tau
C1n = 1/(hy**2)

alpha_m = np.zeros(N+1)
betha_m = np.zeros(N+1)
alpha_m[1] = 1   # здесь использованы граничные условия
alpha1_m = alpha_m[1]  
betha1_m = betha_m[1]  # здесь использованы граничные условия 
 
    
#for tj in range(49):    
for tj in range(49):       
    F = np.zeros((N, M, T))
    for m in range(1, M ):  # от 1 до M-1
        for n in range(1, N): # те границы не беру
            F[n,m,0] = (2* ( 1/tau - 1/( hy**2 ) ) * u[n,m,2*tj]  + ( 1/( hy**2 ) ) * ( u[n,m-1,2*tj] +  u[n,m+1,2*tj] )  + np.cos(x[n]) * np.cos(y[m]) * np.exp(-tau*(tj+1/2)))


    for g in range(1,M):     #для каждого фиксированного m (или i2 из методички)   
        for h in range(1, N): #альфа и бета нулевые не учавствуют, сделано для нумерации (удобства)
            alpha[h + 1] = An / ( Bn - Cn * alpha[h] ) #хотя это можно вынести из цикла g
            betha[h+1] = ( betha[h] * Cn + F[h,g,0] ) / ( Bn - alpha[h] * Cn )

    
        u[N,g,2*tj+1] = betha[N] / ( 1 - alpha[N] ) # здесь использованы граничные условия, это uN

        for q in reversed(range(1,N+1)): 
            u[q-1,g,2*tj+1] = alpha[q] * u[q,g,2*tj+1] + betha[q] # u[q-1,1, 2k +1] должно быть если цикл




 



    F1 = np.zeros((N, M, T))
    for m1 in range(1, M ):  # от 1 до M-1
        for n1 in range(1, N): # те границы не беру
            F1[n1,m1,0] = (2* ( 1/tau - 1/( hx**2 ) ) * u[n1,m1,2*tj+1]  + ( 1/( hx**2 ) ) * ( u[n1-1,m1,2*tj+1] +  u[n1+1,m1,2*tj+1] )  + np.cos(x[n1]) * np.cos(y[m1]) * np.exp(-tau*(2*tj+1)))
        

        
    for g1 in range(1,N):     #для каждого фиксированного m (или i2 из методички)   
        for h1 in range(1, M): #альфа и бета нулевые не учавствуют, сделано для нумерации (удобства)
            alpha_m[h1 + 1] = A1n / ( B1n - C1n * alpha_m[h1] ) #хотя это можно вынести из цикла g
            betha_m[h1+1] = ( betha_m[h1] * C1n + F1[h1,g1,0] ) / ( B1n - alpha_m[h1] * C1n )

    
        u[g1,M,2 + 2*tj] = betha_m[M] / ( 1 - alpha_m[M] ) # здесь использованы граничные условия, это uM

        for q1 in reversed(range(1,M+1)): 
            u[g1,q1-1,2 + 2*tj] = alpha_m[q1] * u[g1,q1,2+2*tj] + betha_m[q1] # u[q-1,1, 2k +1] должно быть если цикл
        u[0,:,2 + 2*tj] = u[1,:,2 + 2*tj]   # зная граничные условия, приравниваем нулевые "компоненты" к "первым"
        u[N,:,2 + 2*tj] = u[N-1,:,2 + 2*tj]  # зная граничные условия, приравниваем последние "компоненты" к "предпоследним"


st.markdown('<h3>Изменение параметров</h3>', unsafe_allow_html=True)

t_ = st.slider('индекс 2*tj + 1, время t = (шаг по времени)*(2*tj + 1)', min_value= 0 , max_value=98, value = 2,  step= 2  )
r1_ = st.slider('вращение по z', min_value= 0 , max_value=90, step=5, value = 20, key="1" )
r2_ = st.slider('вращение в плоскости xy', min_value= -90 , max_value=90, step=5, value = -70, key="2" )


X, Y = np.meshgrid(x, y)
ax = plt.axes(projection='3d')
surff = ax.plot_surface(X, Y, u[:,:,t_], rstride=1, cstride=1, linewidth=0, antialiased=False,
cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('численное решение')
ax.view_init(r1_, r2_ )
ax.set_xticks([i for i in range(4)])
ax.set_yticks([i for i in range(4)])

ax.set_zticks([i - 1 for i in range(3)])
ax.zaxis.set_rotate_label(False)

figure1 = plt.gcf() #захват
st.pyplot(fig = figure1)
plt.clf() #очищение 







#график для ошибки 
st.markdown('<h3>График для ошибок (|аналитическое решение - численное решение|):</h3>', unsafe_allow_html=True)
r1_1 = st.slider('вращение по z', min_value= 0 , max_value=90, step=5, value = 20, key="4" )
r2_2 = st.slider('вращение в плоскости xy', min_value= -90 , max_value=90, step=5, value = -70, key="5" )
st.markdown('<h4>*Вначале нужно убедиться, что времена для численного и аналитического решения синхронизированы (t = (шаг по времени) * (2*tj + 1)).</h4>', unsafe_allow_html=True)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, abs(Z - u[:,:,t_]), rstride=1, cstride=1, linewidth=0, antialiased=False,
cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Аналитическое решение - Численное решение')
ax.view_init(r1_1, r2_2)
ax.set_xticks([i for i in range(4)])
ax.set_yticks([i for i in range(4)])

ax.set_zticks([i - 1 for i in range(3)])
ax.zaxis.set_rotate_label(False)

figure2 = plt.gcf() #захват
st.pyplot(fig = figure2)



st.markdown("<h2><a name=\"top5\">5. Итоги работы </a></h2>", unsafe_allow_html=True)

st.markdown('<p style="text-indent: 25px;"> В данной работе \
обсуждалась краевая задача для уравнения теплопроводности, ее численное и аналитическое \
решения, построенны соответствующие графики. С учетом возможности менять \
параметры системы можно установить, например, зависимость погрешности от шага сетки.\
 Так как в данной работе применялась схема с безусловной устойчивостью, то \
при любых параметрах системы ошибка не сильно возрастала.  <p>  ', unsafe_allow_html=True) 

st.markdown("<h2><a name=\"top6\">6. Приложение. Код </a></h2>", unsafe_allow_html=True)


st.markdown('<p style="text-indent: 25px;"> Здесь представлен код на питоне с комментариями для аналитического \
и численного решений. Вначале аналитическое решение:  <p>  ', unsafe_allow_html=True) 




code = ''' 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import special
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits import mplot3d


t = 0.25

fig = plt.figure()
ax = plt.axes(projection='3d')


def f(x, y, t):
    return np.cos(x) * np.cos(y) * np.exp(-t)


x = np.linspace(0, np.pi, 101)
y = np.linspace(0, np.pi, 101)

X, Y = np.meshgrid(x, y)
Z = f(X, Y, t)

ax = plt.axes(projection='3d')
surff = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False,
cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('Аналитическое решение')
ax.set_xticks([i for i in range(4)])
ax.set_yticks([i for i in range(4)])

ax.set_zticks([i - 1 for i in range(3)])
ax.zaxis.set_rotate_label(False)
plt.show()

'''
st.code(code, language='python')

st.markdown('<p style="text-indent: 25px;"> Ниже представлен код для численного решения: <p>  ', unsafe_allow_html=True) 



code = ''' 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import special
import matplotlib.pyplot as plt

import matplotlib as mpl
from mpl_toolkits import mplot3d


tau = 0.05
k = 0
a = 0.; b = np.pi; c = np.pi; t = 3
u_left = 4.; u_right = -3.5
 # Определение числа интервалов сетки,
 # на которой будет искаться приближённое решение
N = 100
M = 100
hx = b/N
hy = c/M
T = 100
tk = k*tau #но в итоге это не пригодилось
 # Определение cетки
x = np.linspace(a,b,N+1)
y = np.linspace(a,c,M+1)
t = np.linspace(a,t,T)



u = np.zeros((N+1,M+1,T))
for j in range(M+1):
    for i in range(N+1):
        u[i,j,0] = np.cos(x[i])*np.cos(y[j]) #начальные условия

        
An = 1/ (hx**2)
Bn = (2/(hx**2)) + 2/tau
Cn = 1/(hx**2)

alpha = np.zeros(N+1)
betha = np.zeros(N+1)
alpha[1] = 1   # здесь использованы граничные условия
alpha1 = alpha[1]  
betha1 = betha[1]  # здесь использованы граничные условия        



A1n = 1/ (hy**2)
B1n = (2/(hy**2)) + 2/tau
C1n = 1/(hy**2)

alpha_m = np.zeros(N+1)
betha_m = np.zeros(N+1)
alpha_m[1] = 1   # здесь использованы граничные условия
alpha1_m = alpha_m[1]  
betha1_m = betha_m[1]  # здесь использованы граничные условия 
 
    
#for tj in range(49):    
for tj in range(49):       
    F = np.zeros((N, M, T))
    for m in range(1, M ):  # от 1 до M-1
        for n in range(1, N): # те границы не беру
            F[n,m,0] = (2* ( 1/tau - 1/( hy**2 ) ) * u[n,m,2*tj]  + ( 1/( hy**2 ) ) * ( u[n,m-1,2*tj] +  u[n,m+1,2*tj] )  + np.cos(x[n]) * np.cos(y[m]) * np.exp(-tau*(tj+1/2)))
          #в заполнении массива F третью (временную координату) можно было не брать

    for g in range(1,M):     #для каждого фиксированного m (или i2 из методички)   
        for h in range(1, N): #альфа и бета нулевые не учавствуют, сделано для нумерации (удобства)
            alpha[h + 1] = An / ( Bn - Cn * alpha[h] ) #хотя это можно вынести из цикла g
            betha[h+1] = ( betha[h] * Cn + F[h,g,0] ) / ( Bn - alpha[h] * Cn )

    
        u[N,g,2*tj+1] = betha[N] / ( 1 - alpha[N] ) # здесь использованы граничные условия, это uN

        for q in reversed(range(1,N+1)): 
            u[q-1,g,2*tj+1] = alpha[q] * u[q,g,2*tj+1] + betha[q] # u[q-1,1, 2k +1] должно быть если цикл



#Найдя вспомогательное решение, решаем вторую систему

 



    F1 = np.zeros((N, M, T))
    for m1 in range(1, M ):  # от 1 до M-1
        for n1 in range(1, N): # те границы не беру
            F1[n1,m1,0] = (2* ( 1/tau - 1/( hx**2 ) ) * u[n1,m1,2*tj+1]  + ( 1/( hx**2 ) ) * ( u[n1-1,m1,2*tj+1] +  u[n1+1,m1,2*tj+1] )  + np.cos(x[n1]) * np.cos(y[m1]) * np.exp(-tau*(2*tj+1)))
        

        
    for g1 in range(1,N):     #для каждого фиксированного m (или i2 из методички)   
        for h1 in range(1, M): #альфа и бета нулевые не учавствуют, сделано для нумерации (удобства)
            alpha_m[h1 + 1] = A1n / ( B1n - C1n * alpha_m[h1] ) #хотя это можно вынести из цикла g
            betha_m[h1+1] = ( betha_m[h1] * C1n + F1[h1,g1,0] ) / ( B1n - alpha_m[h1] * C1n )

    
        u[g1,M,2 + 2*tj] = betha_m[M] / ( 1 - alpha_m[M] ) # здесь использованы граничные условия, это uM

        for q1 in reversed(range(1,M+1)): 
            u[g1,q1-1,2 + 2*tj] = alpha_m[q1] * u[g1,q1,2+2*tj] + betha_m[q1] # u[q-1,1, 2k +1] должно быть если цикл
        u[0,:,2 + 2*tj] = u[1,:,2 + 2*tj]   # зная граничные условия, приравниваем нулевые "компоненты" к "первым"
        u[N,:,2 + 2*tj] = u[N-1,:,2 + 2*tj]  # зная граничные условия, приравниваем последние "компоненты" к "предпоследним"





X, Y = np.meshgrid(x, y)
ax = plt.axes(projection='3d')
surff = ax.plot_surface(X, Y, u[:,:,4], rstride=1, cstride=1, linewidth=0, antialiased=False,
cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('численное решение')
ax.set_xticks([i for i in range(4)])
ax.set_yticks([i for i in range(4)])

ax.set_zticks([i - 1 for i in range(3)])
ax.zaxis.set_rotate_label(False)
plt.show()

'''
st.code(code, language='python')


st.markdown('<p style="text-indent: 25px;"> График для погрешности: <p>  ', unsafe_allow_html=True) 


code = '''X, Y = np.meshgrid(x, y)
ax = plt.axes(projection='3d')
surff = ax.plot_surface(X, Y, abs(Z - u[:,:,4]), rstride=1, cstride=1, linewidth=0, antialiased=False,
cmap='inferno')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.set_title('аналитическое решение - численное решение')
ax.set_xticks([i for i in range(4)])
ax.set_yticks([i for i in range(4)])

ax.set_zticks([i - 1 for i in range(3)])
ax.zaxis.set_rotate_label(False)
plt.show()'''
st.code(code, language='python')




