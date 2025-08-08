from normalize import normalize_env, normalize_left_right, normalize_symbol


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

# test_normalize_env()