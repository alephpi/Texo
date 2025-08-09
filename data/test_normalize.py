from normalize import (
    normalize_env,
    normalize_expression,
    normalize_left_right,
    normalize_scope,
    normalize_symbol,
)


def test_normalize_env():
    i = r"{ \boldsymbol \mu } _ { \alpha , \alpha ^ { \prime } } = e \int _ { - \infty } ^ { \infty } d { \bf r } \Phi _ { u , \alpha } ( { \bf r } ) ( { \bf r } - { \bf R } _ { u } ) \Phi _ { u , \alpha ^ { \prime } } ( { \bf r } )"
    t = r"{ \boldsymbol \mu } _ { \alpha , \alpha ^ { \prime } } = e \int _ { - \infty } ^ { \infty } d \mathbf { r } \Phi _ { u , \alpha } ( \mathbf { r } ) ( \mathbf { r } - \mathbf { R } _ { u } ) \Phi _ { u , \alpha ^ { \prime } } ( \mathbf { r } )"
    o = normalize_env(i)
    assert o == t

def test_normalize_symbol():
    i = r"\' o \" n \v k | \vert \left \| x"
    t = r"\acute o \ddot n \check k | | \left \| x"
    o = ' '.join(normalize_symbol(i.split()))
    assert o == t

def test_normalize_left_right():
    i = r"x \left( y \right) + z \leftarrow w \rightarrow v"
    t = r"x \left ( y \right ) + z \leftarrow w \rightarrow v"
    o = ' '.join(normalize_left_right(i.split()))
    assert o == t

def test_normalize_expression():
    i = r"k ^ { \prime } m ^ { ' }"
    t = r"k ' m '"
    o = normalize_expression(i)
    assert o == t

def test_normalize_scope():
    i = r"^ { \phantom { {} {\dagger} {} } } \dag ^ { \phantom { \dagger } } \dag"
    t = r"^ { } \dag ^ { } \dag"
    i2 = r"C _ { 1 0 } = \frac { 1 } { 2 } \sqrt { \left( \phantom { } _ { 0 } T _ { 2 1 0 } ^ { c } \right) ^ { 2 } + \left( \phantom { } _ { 0 } T _ { 2 1 0 } ^ { s } \right) ^ { 2 } } ,"
    t2 = r"C _ { 1 0 } = \frac { 1 } { 2 } \sqrt { \left( _ { 0 } T _ { 2 1 0 } ^ { c } \right) ^ { 2 } + \left( _ { 0 } T _ { 2 1 0 } ^ { s } \right) ^ { 2 } } ,"
    i3 = r"\phantom { } _ { 1 } \Delta \bar { C } _ { 3 0 }"
    t3 = r"_ { 1 } \Delta \bar { C } _ { 3 0 }"
    i4 = r"\phantom { }"
    t4 = r""
    i5 = r"^ { \vphantom \dagger }"
    t5 = r"^ { }"

    o = ' '.join(normalize_scope(i.split()))
    o2 = ' '.join(normalize_scope(i2.split()))
    o3 = ' '.join(normalize_scope(i3.split()))
    o4 = ' '.join(normalize_scope(i4.split()))
    o5 = ' '.join(normalize_scope(i5.split()))
    assert o == t
    assert o2 == t2
    assert o3 == t3
    assert o4 == t4
    assert o5 == t5
