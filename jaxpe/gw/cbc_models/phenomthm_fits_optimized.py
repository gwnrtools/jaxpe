import jax
import jax.numpy as jnp


@jax.jit
def compute_all_phenomthm_fits(eta, S, dchi, delta):
    """
    Computes all phenomenological fits simultaneously.
    This leverages XLA's Common Subexpression Elimination (CSE) to massively
    speed up the evaluation of polynomial terms (like eta^2, S^3, etc.) shared across fits.
    """
    fits = {}
    fits["IMRPhenomT_PeakFrequency_22"] = (
        0.27212130745330404
        + 0.40972689759932074 * eta
        - 0.0018392172960247433 * eta * jnp.power(dchi, 2)
        + S
        * (
            0.09558832959428547
            - 0.04834585264918328 * eta
            - 0.15275173823699056 * jnp.power(eta, 2)
        )
        - 3.4232387074402153 * jnp.power(eta, 2)
        + 32.853772442252605 * jnp.power(eta, 3)
        - 1.4976829186605336
        * dchi
        * delta
        * (1 - 4.775645585721007 * eta)
        * jnp.power(eta, 3)
        - 0.9981117852179613
        * dchi
        * delta
        * (1 - 5.260098925354571 * eta)
        * S
        * jnp.power(eta, 3)
        - 125.22505746137587 * jnp.power(eta, 4)
        + 179.3797198714914 * jnp.power(eta, 5)
        + (
            0.054391696704622204
            - 0.1482682698299456 * eta
            + 0.08938162810617255 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            -0.020719540055375383
            + 0.5090144456500953 * eta
            - 1.5809441589349338 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + (
            0.024240736699062685
            - 0.09490089674418004 * eta
            + 0.09518501714836035 * jnp.power(eta, 2)
        )
        * jnp.power(S, 4)
        + (
            0.09759303647532228
            - 1.105520690228567 * eta
            + 2.921271981239294 * jnp.power(eta, 2)
        )
        * jnp.power(S, 5)
    )
    fits["IMRPhenomT_RD_Freq_D2_22"] = (
        0.1598180460429256
        + 0.19120040104567676 * eta
        + (-0.012853620630980167 - 0.006532392920798404 * eta) * S
        - 0.7733759581766899 * jnp.power(eta, 2)
        + 0.18151402648790957
        * dchi
        * delta
        * (1 - 9.041198282315879 * eta)
        * jnp.power(eta, 2)
        + 0.27147713896183995
        * dchi
        * delta
        * (1 - 5.653323210961101 * eta)
        * S
        * jnp.power(eta, 2)
        - 0.01603489049446065 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (-0.046785083372074494 + 0.102759380109996 * eta) * jnp.power(S, 2)
        + (0.0009883572415502464 - 0.050384608002279486 * eta) * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Freq_D3_22"] = (
        2.6456463496860927
        - 28.079375863863458 * eta
        + 323.1691069138812 * jnp.power(eta, 2)
        - 0.5040057675360762
        * dchi
        * delta
        * (1 + 21.786482297795278 * eta)
        * jnp.power(eta, 2)
        + 1.561247215701216
        * dchi
        * delta
        * (1 - 1.7508069810164308 * eta)
        * S
        * jnp.power(eta, 2)
        + S
        * (
            3.091917073632116
            - 17.345283345692266 * eta
            + 33.40735388809028 * jnp.power(eta, 2)
        )
        - 1490.8128941604907 * jnp.power(eta, 3)
        + 0.1619056474567525 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + 2376.3257196613886 * jnp.power(eta, 4)
        + (
            0.734022429223849
            - 0.029342234233198747 * eta
            - 9.281610698291932 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP1_22"] = (
        0.00006480771730217768 * eta * jnp.power(dchi, 2)
        - 0.3543965558027252
        * dchi
        * delta
        * (1 - 2.463526130684083 * eta)
        * jnp.power(eta, 3)
        + 0.01879295038873938
        * dchi
        * delta
        * (1 - 5.236796607517272 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.1472653807120573 * eta
            - 1.9636752493349356 * jnp.power(eta, 2)
            + 14.177521724634461 * jnp.power(eta, 3)
            - 48.94620901701877 * jnp.power(eta, 4)
            + 63.83730899015984 * jnp.power(eta, 5)
        )
        + eta
        * (
            0.8493442097893826
            - 13.211067914003836 * eta
            + 311.99021467938235 * jnp.power(eta, 2)
            - 4731.025904601601 * jnp.power(eta, 3)
            + 44821.93042533854 * jnp.power(eta, 4)
            - 264474.1374080295 * jnp.power(eta, 5)
            + 943246.2317701122 * jnp.power(eta, 6)
            - 1.8588135904328802e6 * jnp.power(eta, 7)
            + 1.5524778581809246e6 * jnp.power(eta, 8)
        )
        + (
            0.04902976057622393 * eta
            - 1.0152511131279736 * jnp.power(eta, 2)
            + 8.286289152216145 * jnp.power(eta, 3)
            - 30.19775956110767 * jnp.power(eta, 4)
            + 40.670065442751955 * jnp.power(eta, 5)
        )
        * jnp.power(S, 2)
        + (
            0.04780630695082567 * eta
            - 1.2177827888317065 * jnp.power(eta, 2)
            + 11.505675146308567 * jnp.power(eta, 3)
            - 46.733420749352135 * jnp.power(eta, 4)
            + 68.40821782168776 * jnp.power(eta, 5)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP2_22"] = (
        0.000100027278976821 * eta * jnp.power(dchi, 2)
        - 0.7578403155712378
        * dchi
        * delta
        * (1 - 2.056456271350877 * eta)
        * jnp.power(eta, 3)
        - 0.14126282637778914
        * dchi
        * delta
        * (1 - 2.5840771007494916 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.2331970217833686 * eta
            - 1.5473968380422929 * jnp.power(eta, 2)
            + 5.973401506474942 * jnp.power(eta, 3)
            - 9.110484789161045 * jnp.power(eta, 4)
        )
        + eta
        * (
            0.9904613241626621
            - 6.708006572605403 * eta
            + 127.40270095439482 * jnp.power(eta, 2)
            - 1723.355339710798 * jnp.power(eta, 3)
            + 15430.10086310527 * jnp.power(eta, 4)
            - 88744.26044058547 * jnp.power(eta, 5)
            + 313650.01696201024 * jnp.power(eta, 6)
            - 617887.8122937253 * jnp.power(eta, 7)
            + 518220.9267888211 * jnp.power(eta, 8)
        )
        + (
            0.08934817374146888 * eta
            - 0.8887847358339216 * jnp.power(eta, 2)
            + 3.7233864099350784 * jnp.power(eta, 3)
            - 5.814765403882651 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            0.04471990627820145 * eta
            - 0.642458648615624 * jnp.power(eta, 2)
            + 3.393481171493086 * jnp.power(eta, 3)
            - 6.092083983738554 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP3_22"] = (
        0.0002459376633671657 * eta * jnp.power(dchi, 2)
        - 0.8794763631110696
        * dchi
        * delta
        * (1 - 2.0751630535350096 * eta)
        * jnp.power(eta, 3)
        - 0.3319387797134261
        * dchi
        * delta
        * (1 - 3.1838055629892184 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.23505507416274007 * eta
            - 1.2449030421324767 * jnp.power(eta, 2)
            + 4.315803728759738 * jnp.power(eta, 3)
            - 6.384257606413192 * jnp.power(eta, 4)
        )
        + eta
        * (
            1.0208762064809185
            - 3.3799457394243957 * eta
            + 16.242639717123314 * jnp.power(eta, 2)
            + 299.2297416582362 * jnp.power(eta, 3)
            - 5913.920743907752 * jnp.power(eta, 4)
            + 46388.231537995445 * jnp.power(eta, 5)
            - 192261.0498470111 * jnp.power(eta, 6)
            + 413750.14250475995 * jnp.power(eta, 7)
            - 364403.84935539874 * jnp.power(eta, 8)
        )
        + (
            0.09630827896641526 * eta
            - 0.7915321134872877 * jnp.power(eta, 2)
            + 2.86907420250287 * jnp.power(eta, 3)
            - 4.038995403653199 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            0.07395420485618898 * eta
            - 1.0289224187583748 * jnp.power(eta, 2)
            + 5.275845823734598 * jnp.power(eta, 3)
            - 9.206158044409037 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Merger_Amp_CP1_22"] = (
        0.0004059354652663733 * eta * jnp.power(dchi, 2)
        - 0.9382383412276684
        * dchi
        * delta
        * (1 - 2.509151362054917 * eta)
        * jnp.power(eta, 3)
        - 0.6560748977864668
        * dchi
        * delta
        * (1 - 3.426294113321932 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.23465398091766254 * eta
            - 1.3398914201113978 * jnp.power(eta, 2)
            + 5.9073801933446495 * jnp.power(eta, 3)
            - 10.84221896204708 * jnp.power(eta, 4)
        )
        + eta
        * (
            1.2946032382158479
            - 3.3343035556341816 * eta
            + 91.6430240976277 * jnp.power(eta, 2)
            - 1687.6195123629968 * jnp.power(eta, 3)
            + 19726.50907350641 * jnp.power(eta, 4)
            - 140798.18973779568 * jnp.power(eta, 5)
            + 594095.3303894227 * jnp.power(eta, 6)
            - 1.358657562562124e6 * jnp.power(eta, 7)
            + 1.2958912179017465e6 * jnp.power(eta, 8)
        )
        + (
            0.03174875260265387 * eta
            + 0.23082150180902375 * jnp.power(eta, 2)
            - 1.9901867982613048 * jnp.power(eta, 3)
            + 4.009389679757772 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            -0.04033221614773138 * eta
            + 0.8426888041517518 * jnp.power(eta, 2)
            - 4.742283264846479 * jnp.power(eta, 3)
            + 9.059923021547936 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_PeakAmp_22"] = (
        0.0017885007700308166 * eta * jnp.power(dchi, 2)
        - 0.5846280668038513
        * dchi
        * delta
        * (1 - 4.879882766464646 * eta)
        * jnp.power(eta, 3)
        - 0.874161608112943
        * dchi
        * delta
        * (1 - 1.690095043235707 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.203557188205307 * eta
            - 2.4368458739010563 * jnp.power(eta, 2)
            + 12.206344183078137 * jnp.power(eta, 3)
            - 23.417979354674692 * jnp.power(eta, 4)
        )
        + eta
        * (
            1.4701266133411792
            - 1.387711607537906 * eta
            + 25.641251409467607 * jnp.power(eta, 2)
            - 186.013359336165 * jnp.power(eta, 3)
            + 801.3039484150348 * jnp.power(eta, 4)
            - 1893.8181854645718 * jnp.power(eta, 5)
            + 1946.531703997353 * jnp.power(eta, 6)
        )
        + (
            -0.0018659293826992745 * eta
            - 0.1888206507658455 * jnp.power(eta, 2)
            + 1.4677324802664107 * jnp.power(eta, 3)
            - 1.4019283350536489 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            -0.14699838946027494 * eta
            + 2.6186847787143837 * jnp.power(eta, 2)
            - 15.574381075605208 * jnp.power(eta, 3)
            + 31.239292792717016 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Amp_C3_22"] = (
        -0.48053994718185694
        + 0.7023672141561462 * eta
        + S
        * (
            -0.3597773028596323
            + 1.4330280386796503 * eta
            - 3.239121799338561 * jnp.power(eta, 2)
        )
        - 0.1993836305574211 * jnp.power(eta, 2)
        + (
            -0.2651107472061685
            + 1.6433443489711386 * eta
            - 2.757772023954491 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            -0.01973537883495192
            - 0.2410762147438714 * eta
            + 2.7315015976869756 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Merger_Freq_CP1_21"] = (
        0.10101116560411222
        - 0.0018259335648908525 * eta
        - 0.021099940258397783 * jnp.power(eta, 2)
        - 0.20805682832251896
        * dchi
        * delta
        * (1 - 2.941803484525313 * eta)
        * jnp.power(eta, 2)
        - 0.34936387567469845
        * dchi
        * delta
        * (1 - 3.7803158001441313 * eta)
        * S
        * jnp.power(eta, 2)
        + S
        * (
            0.04417410903550135
            - 0.043497437478068064 * eta
            - 0.1259814374130449 * jnp.power(eta, 2)
            + 0.32193272002103757 * jnp.power(eta, 3)
        )
        + 1.5195639445091516 * jnp.power(eta, 3)
        + 0.027556221044750074 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        - 3.734211998389987 * jnp.power(eta, 4)
        + (
            0.018056562830041197
            + 0.18034946569991206 * eta
            - 1.6734011535036983 * jnp.power(eta, 2)
            + 3.234630062543167 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.04137468913747541
            - 0.3769952802830424 * eta
            + 1.220244052602933 * jnp.power(eta, 2)
            - 1.1481788805520008 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
        + (
            0.04441609678703693
            - 0.5527880503765283 * eta
            + 1.9248039734409312 * jnp.power(eta, 2)
            - 1.2987670735175931 * jnp.power(eta, 3)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_Merger_Freq_CP1_33"] = (
        0.28925318007299916
        - 0.11957063600442912 * eta
        + 1.1589911273564353 * jnp.power(eta, 2)
        - 0.5263424292081523
        * dchi
        * delta
        * (1 - 1.8018453981168454 * eta)
        * jnp.power(eta, 2)
        - 1.509871876079703
        * dchi
        * delta
        * (1 - 4.654249984146787 * eta)
        * S
        * jnp.power(eta, 2)
        - 1.5455016775261783 * jnp.power(eta, 3)
        - 0.08410864613712594 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            0.13053774656007028
            - 0.1436725066031307 * eta
            - 0.3988589708046106 * jnp.power(eta, 2)
            + 1.6814308803706919 * jnp.power(eta, 3)
        )
        + (
            0.06962116377895124
            - 0.25606130995761806 * eta
            + 2.809259385656226 * jnp.power(eta, 2)
            - 10.655152499726746 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.13941546691342796
            - 1.4663373405231797 * eta
            + 7.242184758809972 * jnp.power(eta, 2)
            - 14.156137151704971 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
        + (
            0.11741728765640543
            - 0.9473051744486624 * eta
            + 0.1981448121599412 * jnp.power(eta, 2)
            + 8.2668429483671 * jnp.power(eta, 3)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_Merger_Freq_CP1_44"] = (
        0.3839588385106795
        - 0.1433725509161493 * eta
        + S
        * (
            0.16172235324877973
            - 0.029039848526679315 * eta
            - 0.5078615524759549 * jnp.power(eta, 2)
        )
        + 1.549008100266779 * jnp.power(eta, 2)
        - 2.782419405121844 * jnp.power(eta, 3)
        - 6.300899329929631
        * dchi
        * delta
        * (1 - 3.4163840266407193 * eta)
        * jnp.power(eta, 3)
        + 3.785702496330771e-7
        * dchi
        * delta
        * (1 - 7.683429533133828e6 * eta)
        * S
        * jnp.power(eta, 3)
        - 0.006896021141579361 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (
            0.08448608215066597
            - 0.04225970075410604 * eta
            - 0.5752978186367078 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            0.19977344584848072
            - 1.6500083087892263 * eta
            + 3.6200398572688224 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + (
            0.17526054184412337
            - 1.6567766073748962 * eta
            + 3.920825944654624 * jnp.power(eta, 2)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_Merger_Freq_CP1_55"] = (
        0.49157926097800314
        - 0.5109882379431707 * eta
        + S
        * (
            0.23504290986387008
            - 0.4663579238504208 * eta
            + 0.6335674200647748 * jnp.power(eta, 2)
        )
        + 5.1971222617574755 * jnp.power(eta, 2)
        - 17.312514984111704 * jnp.power(eta, 3)
        - 8.220653931136855
        * dchi
        * delta
        * (1 - 3.377023099309163 * eta)
        * jnp.power(eta, 3)
        - 4.92651904204219
        * dchi
        * (1 - 4.0976182182508625 * eta)
        * S
        * jnp.power(eta, 3)
        + 22.250087206692136 * jnp.power(eta, 4)
        + (
            0.01818932761340069
            + 1.2525110439050509 * eta
            - 4.785925497931221 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            0.05473740403349404
            + 0.204944664491078 * eta
            - 1.5320283880617676 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + (
            0.19945251844606834
            - 2.3208613314645397 * eta
            + 6.551836762093784 * jnp.power(eta, 2)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_PeakFrequency_21"] = (
        0.17642087831932626
        + 0.31718290537914057 * eta
        + S
        * (
            0.03094734575888092
            + 0.07319676429288274 * eta
            - 0.4370939605469398 * jnp.power(eta, 2)
        )
        - 2.2156624517537873 * jnp.power(eta, 2)
        + 14.007580103948815 * jnp.power(eta, 3)
        + 2.641340486447181
        * dchi
        * delta
        * (1 - 6.221406704917193 * eta)
        * jnp.power(eta, 3)
        + 4.353108475005447
        * dchi
        * delta
        * (1 - 8.473808274993978 * eta)
        * S
        * jnp.power(eta, 3)
        - 0.036084481180729745 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        - 26.085064860873068 * jnp.power(eta, 4)
        + (
            0.017462707546942863
            - 0.11463071986182106 * eta
            + 0.2800463367551972 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            0.033646159761323895
            - 0.33812286198554814 * eta
            + 0.9635090140454092 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + (
            -0.0011779170821489254
            + 0.18369948603536548 * eta
            - 0.5978006007616697 * jnp.power(eta, 2)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_PeakFrequency_33"] = (
        0.42535721148121036
        + 0.3085253521281911 * eta
        + S
        * (
            0.1280277017287708
            + 0.15271593642827125 * eta
            - 0.9083681800119519 * jnp.power(eta, 2)
        )
        + 0.9392741497311157 * jnp.power(eta, 2)
        - 0.20785772397714286 * dchi * (1 - 3.487216886252809 * eta) * jnp.power(eta, 2)
        - 0.7863911902548658
        * dchi
        * (1 - 4.74913840513059 * eta)
        * S
        * jnp.power(eta, 2)
        - 0.41376935975085416 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (
            0.09308538633777035
            + 0.055164833113211194 * eta
            - 1.1480525120934546 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            0.09945668702979882
            - 0.5488068825374101 * eta
            + 0.8675986447602085 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_PeakFrequency_44"] = (
        0.5640094664638
        + 0.3956446752668519 * eta
        + S
        * (
            0.16597514208305744
            + 0.38143981208933403 * eta
            - 1.9002920053147696 * jnp.power(eta, 2)
        )
        + 2.5091004914938675 * jnp.power(eta, 2)
        - 7.403354368373608 * jnp.power(eta, 3)
        - 5.257927939622048
        * dchi
        * delta
        * (1 - 5.385135507412752 * eta)
        * jnp.power(eta, 3)
        + 1.1110261817411248e-7
        * dchi
        * delta
        * (1 - 1.0881293779054403e7 * eta)
        * S
        * jnp.power(eta, 3)
        - 0.06378504432547372 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (
            0.08205749018653839
            + 0.016449185328805776 * eta
            - 0.509112344628105 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            0.13245468901111399
            - 1.0716792675901017 * eta
            + 2.631350201223915 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + (
            0.0798820256896006
            - 0.6976704383121812 * eta
            + 1.6808658698855679 * jnp.power(eta, 2)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_PeakFrequency_55"] = (
        0.7146297908371999
        + 0.1421128402132339 * eta
        + 7.659311331111322 * jnp.power(eta, 2)
        + S
        * (
            0.29191927041842664
            - 0.6512295551490094 * eta
            + 1.021846701552054 * jnp.power(eta, 2)
        )
        - 38.14301940776831 * jnp.power(eta, 3)
        - 3.460574689440357
        * dchi
        * delta
        * (1 - 4.738903271021608 * eta)
        * jnp.power(eta, 3)
        - 8.262749319140365
        * dchi
        * (1 - 4.1126856272636285 * eta)
        * S
        * jnp.power(eta, 3)
        + 69.0208119373966 * jnp.power(eta, 4)
        + (
            -0.17737667108149985
            + 4.564503709808925 * eta
            - 15.457705511019 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            -0.08755132408422435
            + 1.8185807604067965 * eta
            - 5.710975144545469 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + (
            0.4020378024101137
            - 6.137619764177151 * eta
            + 19.730459568297885 * jnp.power(eta, 2)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_RD_Freq_D2_21"] = (
        0.1781545202005886
        + 0.10906983816039043 * eta
        + (-0.023905104384959013 + 0.1831847458257083 * eta) * S
        - 0.5060291743082528 * jnp.power(eta, 2)
        - 0.309304704734991
        * dchi
        * delta
        * (1 - 2.5742929128570724 * eta)
        * jnp.power(eta, 2)
        + 2.1883684085193034
        * dchi
        * delta
        * (1 - 4.850311934387953 * eta)
        * S
        * jnp.power(eta, 2)
        + 0.25978316114962485 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (-0.00955976176018747 - 0.18697585595061622 * eta) * jnp.power(S, 2)
        + (0.04468930365659441 - 0.44170842157754653 * eta) * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Freq_D3_21"] = (
        3.757258772469613
        + 0.08380574896251641 * eta
        + (3.634895503051922 - 8.174660936683596 * eta) * S
        - 18.17832018576314 * jnp.power(eta, 2)
        + 0.000043786944413372623
        * dchi
        * delta
        * (1 - 2.0348094407156347e6 * eta)
        * jnp.power(eta, 2)
        + 60.2475724845033
        * dchi
        * delta
        * (1 - 6.868024913964549 * eta)
        * S
        * jnp.power(eta, 2)
        + 23.08504961982195 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (3.7989582331116707 - 19.36029310028481 * eta) * jnp.power(S, 2)
    )
    fits["IMRPhenomT_RD_Freq_D2_33"] = (
        0.16417885317959574
        + 0.25804336274633655 * eta
        + (-0.02961300365618534 - 0.006664043292875596 * eta) * S
        - 0.9792038927762032 * jnp.power(eta, 2)
        + 1.9953234313463062
        * dchi
        * delta
        * (1 - 5.45249062802972 * eta)
        * jnp.power(eta, 2)
        + 6.3956780147142e-6
        * dchi
        * delta
        * (1 + 1.617071409433673e6 * eta)
        * S
        * jnp.power(eta, 2)
        - 1.2001578283095737 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (-0.06346953330083285 + 0.12623926538220964 * eta) * jnp.power(S, 2)
        + (-0.015173742568790456 + 0.016604992725771543 * eta) * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Freq_D3_33"] = (
        2.0503935647397173
        + 2.238118245943281 * eta
        + (0.7121733508300451 - 0.397525795105057 * eta) * S
        - 12.794117052655967 * jnp.power(eta, 2)
        + 36.20571065481121
        * dchi
        * delta
        * (1 - 5.537022415039092 * eta)
        * jnp.power(eta, 2)
        + 0.00018637091124974205
        * dchi
        * delta
        * (1 + 1.2667084084427443e6 * eta)
        * S
        * jnp.power(eta, 2)
        - 21.894760631998928 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (0.4988930825119534 - 3.4004257158793045 * eta) * jnp.power(S, 2)
        + (1.0586608433869105 - 5.625073332864818 * eta) * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Freq_D2_44"] = (
        0.21336620664104916
        - 0.20527614713716544 * eta
        + (-0.057793617403743454 + 0.234794019739202 * eta) * S
        - 0.5040007874429419 * jnp.power(eta, 2)
        + 1.7980712659091223
        * dchi
        * delta
        * (1 - 4.3332243187779715 * eta)
        * jnp.power(eta, 2)
        + 5.615398937364741
        * dchi
        * delta
        * (1 - 4.67655881619209 * eta)
        * S
        * jnp.power(eta, 2)
        + 0.019754287494577062 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (-0.06870289106806035 + 0.18761585848765555 * eta) * jnp.power(S, 2)
    )
    fits["IMRPhenomT_RD_Freq_D3_44"] = (
        3.824722911046124
        - 19.434149952978917 * eta
        + (1.1492216649186293 - 0.1794193390842707 * eta) * S
        + 46.15218473526611 * jnp.power(eta, 2)
        - 5.329235322993467e-6
        * dchi
        * delta
        * (1 + 3.600409478406777e6 * eta)
        * jnp.power(eta, 2)
        + 108.44244729377574
        * dchi
        * delta
        * (1 - 4.948832996449642 * eta)
        * S
        * jnp.power(eta, 2)
        + 0.8202829030161741 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (0.3851268017278357 - 1.4577127768796085 * eta) * jnp.power(S, 2)
    )
    fits["IMRPhenomT_RD_Freq_D2_55"] = (
        0.2143703929690296
        - 0.26905171511199966 * eta
        + (-0.057285673301351384 + 0.22530123030818466 * eta) * S
        - 0.22128464791686953 * jnp.power(eta, 2)
        + 1.2330723562386177
        * dchi
        * delta
        * (1 - 5.234362591591656 * eta)
        * jnp.power(eta, 2)
        + 2.7651387378521104
        * dchi
        * delta
        * (1 - 4.529998650048839 * eta)
        * S
        * jnp.power(eta, 2)
        - 0.02296167789737978 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (-0.06187017465583654 + 0.19469068079404978 * eta) * jnp.power(S, 2)
    )
    fits["IMRPhenomT_RD_Freq_D3_55"] = (
        3.421296704767661
        - 12.809224663506237 * eta
        + (0.17979866256123875 + 3.9220602497543733 * eta) * S
        + 22.579308761647468 * jnp.power(eta, 2)
        + 38.938817471901075
        * dchi
        * delta
        * (1 - 5.439168816882736 * eta)
        * jnp.power(eta, 2)
        + 102.01005958254325
        * dchi
        * delta
        * (1 - 4.94517313697764 * eta)
        * S
        * jnp.power(eta, 2)
        + 8.156992533112646 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (-0.8276383989425874 + 5.653818482979737 * eta) * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP1_21"] = (
        -0.2457309233525402 * dchi * (1 - 1.8588313811238013 * eta) * jnp.power(eta, 3)
        + 0.007720682776232238
        * dchi
        * (1 - 14.5539282402835 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.00002718410442799091 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            -0.019371607120048675 * delta * eta
            + 0.03368798661754525 * delta * jnp.power(eta, 2)
            - 0.0347647962890128 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.12222678288098383
            - 2.152654527154567 * eta
            + 34.53692688859637 * jnp.power(eta, 2)
            - 317.45437636541044 * jnp.power(eta, 3)
            + 1625.665271951051 * jnp.power(eta, 4)
            - 4325.99209923682 * jnp.power(eta, 5)
            + 4661.112076870376 * jnp.power(eta, 6)
        )
        + (
            -0.004130586129052499 * delta * eta
            - 0.034242170459751614 * delta * jnp.power(eta, 2)
            + 0.1845040639852827 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.00023312994425693458 * delta * eta
            - 0.006465524142621246 * delta * jnp.power(eta, 2)
            + 0.02059744168116181 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
        + (
            -0.010994253719930009 * delta * eta
            + 0.1617856319808047 * delta * jnp.power(eta, 2)
            - 0.5128238142456396 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP2_21"] = (
        -0.5514762410690445 * dchi * (1 - 1.6606901062713382 * eta) * jnp.power(eta, 3)
        - 0.021703163232290525
        * dchi
        * (1 + 12.285199361388841 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.00027551818326783677 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            -0.013014289088905106 * delta * eta
            - 0.14836733162360224 * delta * jnp.power(eta, 2)
            + 0.3879852721571224 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.15072063925506032
            - 0.8093028329445506 * eta
            + 6.206684655292913 * jnp.power(eta, 2)
            - 24.88401414398108 * jnp.power(eta, 3)
            + 38.250250718164864 * jnp.power(eta, 4)
        )
        + (
            -0.025960288375186314 * delta * eta
            + 0.09485066561654602 * delta * jnp.power(eta, 2)
            - 0.12985415687429802 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            -0.031051316903933826 * delta * eta
            + 0.29808639962599887 * delta * jnp.power(eta, 2)
            - 0.7880170636799876 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP3_21"] = (
        -0.6553365123485911 * dchi * (1 - 1.5398595374318753 * eta) * jnp.power(eta, 3)
        - 0.03414520050962973
        * dchi
        * (1 + 10.152070659598607 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.0003514981514078436 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            -0.02358276114828079 * delta * eta
            - 0.06676889646672902 * delta * jnp.power(eta, 2)
            + 0.10431702660244097 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.16554873311231985
            - 0.6991328198972108 * eta
            + 4.8998331628863 * jnp.power(eta, 2)
            - 17.811340834192666 * jnp.power(eta, 3)
            + 25.04713555013603 * jnp.power(eta, 4)
        )
        + (
            -0.03170047769861336 * delta * eta
            + 0.12228560605854709 * delta * jnp.power(eta, 2)
            - 0.2157318828663416 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            -0.02241156276655523 * delta * eta
            + 0.1503547988268005 * delta * jnp.power(eta, 2)
            - 0.32463957366468943 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Merger_Amp_CP1_21"] = (
        -0.9639235481813841 * dchi * (1 - 0.0953303705707973 * eta) * jnp.power(eta, 3)
        + 0.023311043707270836
        * dchi
        * (1 - 45.56758470014973 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.0006882033227649262 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            -0.07909979765641208 * delta * eta
            - 0.1082323292489176 * delta * jnp.power(eta, 2)
            - 0.059050965420919255 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.2650216550129358
            - 0.6689867318991676 * eta
            + 6.322130556715582 * jnp.power(eta, 2)
            - 25.30530828638204 * jnp.power(eta, 3)
            + 37.57098989317853 * jnp.power(eta, 4)
        )
        + (
            -0.048774914332679824 * delta * eta
            + 0.058892171955548835 * delta * jnp.power(eta, 2)
            + 0.04519042054728839 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            -0.04972189708933256 * delta * eta
            + 0.32903712470645846 * delta * jnp.power(eta, 2)
            - 0.7095763077249748 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_PeakAmp_21"] = (
        -1.124757115880216 * dchi * (1 + 3.9089731034256547 * eta) * jnp.power(eta, 3)
        + 0.14171442436657175 * dchi * S * jnp.power(eta, 3)
        - 7.996997960509883e-6
        * (1 + 12111.971615981536 * delta)
        * jnp.power(dchi, 2)
        * jnp.power(eta, 3)
        + delta
        * eta
        * (
            0.5940439865028524
            - 2.6802250765521083 * eta
            + 23.43295820742704 * jnp.power(eta, 2)
            - 89.91427919476679 * jnp.power(eta, 3)
            + 129.10731997830192 * jnp.power(eta, 4)
        )
        + S
        * (
            -0.40438488955545776 * delta * eta
            + 0.6359546829540189 * delta * jnp.power(eta, 2)
            - 7.6174781238188 * delta * jnp.power(eta, 3)
            + 20.156475820119724 * delta * jnp.power(eta, 4)
        )
        + (
            -0.04723336574759155 * delta * eta
            + 0.18082387349024776 * delta * jnp.power(eta, 2)
            + 1.7306679608818485 * delta * jnp.power(eta, 3)
            - 8.236553093624009 * delta * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            -0.12534984288882925 * delta * eta
            + 0.6131320823681302 * delta * jnp.power(eta, 2)
            + 5.1648126976659885 * delta * jnp.power(eta, 3)
            - 24.289576920541403 * delta * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
        + (
            0.07112546745185065 * delta * eta
            - 1.3149279454050955 * delta * jnp.power(eta, 2)
            + 8.514263145733384 * delta * jnp.power(eta, 3)
            - 14.271807407363035 * delta * jnp.power(eta, 4)
        )
        * jnp.power(S, 4)
    )
    fits["IMRPhenomT_RD_Amp_C3_21"] = (
        -0.04334302376511826
        - 0.17676752692299327 * eta
        + 0.4505339209591958 * jnp.power(eta, 2)
        + S
        * (
            -0.06491024396051823
            - 1.1215130164808509 * eta
            + 2.5523011435327345 * jnp.power(eta, 2)
        )
        + (
            -0.28713100991035806
            + 1.2391262662740283 * eta
            - 2.7551841346664796 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            -0.6910312848115802
            + 5.91843541910692 * eta
            - 13.892447750204266 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
        + 0 * dchi
    )
    fits["IMRPhenomT_Inspiral_Amp_CP1_33"] = (
        -0.00005000414942937797
        * delta
        * (1 - 3.0430401949925754 * eta)
        * eta
        * jnp.power(dchi, 2)
        - 0.03836271211298855 * dchi * (1 - 4.654767900586748 * eta) * jnp.power(eta, 3)
        + 0.007041962008283751
        * dchi
        * (1 - 3.238646631077093 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.0432725315235326 * delta * eta
            - 0.3128744737439017 * delta * jnp.power(eta, 2)
            + 0.7249180430447414 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.22272167356880285
            - 3.217949139895537 * eta
            + 45.52929729100423 * jnp.power(eta, 2)
            - 379.70414120110206 * jnp.power(eta, 3)
            + 1801.6287410802781 * jnp.power(eta, 4)
            - 4505.468825419055 * jnp.power(eta, 5)
            + 4606.517765490795 * jnp.power(eta, 6)
        )
        + (
            0.015232248632190103 * delta * eta
            - 0.15205944312376768 * delta * jnp.power(eta, 2)
            + 0.38322848961855754 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP2_33"] = (
        -0.0005485061120167634
        * delta
        * (1 - 5.249847868911592 * eta)
        * eta
        * jnp.power(dchi, 2)
        - 0.13406080756104294 * dchi * (1 - 4.791415116248203 * eta) * jnp.power(eta, 3)
        - 0.025192101240368327
        * dchi
        * (1 - 5.557132409376257 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.09903436069097878 * delta * eta
            - 0.5266490647574258 * delta * jnp.power(eta, 2)
            + 1.082646288776612 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.296454036493377
            - 2.741774425959176 * eta
            + 42.10341453030946 * jnp.power(eta, 2)
            - 391.5079943491554 * jnp.power(eta, 3)
            + 2084.7204836711294 * jnp.power(eta, 4)
            - 5857.9923995429735 * jnp.power(eta, 5)
            + 6724.299707693131 * jnp.power(eta, 6)
        )
        + (
            0.04482378193895758 * delta * eta
            - 0.34909482801592684 * delta * jnp.power(eta, 2)
            + 0.798188874585321 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP3_33"] = (
        -0.00014989518553589642
        * delta
        * (1 + 0.10284764229097754 * eta)
        * eta
        * jnp.power(dchi, 2)
        - 0.16531803034216744
        * dchi
        * (1 - 4.9470029202324755 * eta)
        * jnp.power(eta, 3)
        - 0.031723644862959394
        * dchi
        * (1 - 5.870965439700585 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.11070499324391728 * delta * eta
            - 0.5112660954416434 * delta * jnp.power(eta, 2)
            + 0.9943348519498412 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.3081004973876298
            - 1.4982270638091204 * eta
            + 10.664775575232024 * jnp.power(eta, 2)
            + 2.0410986773159214 * jnp.power(eta, 3)
            - 472.97637767340444 * jnp.power(eta, 4)
            + 2442.526427205543 * jnp.power(eta, 5)
            - 3894.4435672165723 * jnp.power(eta, 6)
        )
        + (
            0.0509202148340841 * delta * eta
            - 0.3395424984982766 * delta * jnp.power(eta, 2)
            + 0.7165890644210602 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Merger_Amp_CP1_33"] = (
        -0.0009410965748168944
        * delta
        * (1 - 7.241296835003745 * eta)
        * eta
        * jnp.power(dchi, 2)
        - 0.14546859303533213 * dchi * (1 - 8.38731410081936 * eta) * jnp.power(eta, 3)
        - 0.1398648328352838
        * dchi
        * (1 - 5.537001000046034 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.11682330636277577 * delta * eta
            - 0.2730553632132845 * delta * jnp.power(eta, 2)
            + 0.5237293086635135 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.44186265057020313
            - 1.2636555898615027 * eta
            + 18.126195225020272 * jnp.power(eta, 2)
            - 130.51981907268976 * jnp.power(eta, 3)
            + 526.5238580108073 * jnp.power(eta, 4)
            - 1078.532545666921 * jnp.power(eta, 5)
            + 849.9216826816613 * jnp.power(eta, 6)
        )
        + (
            0.0510403796344345 * delta * eta
            - 0.180522632008535 * delta * jnp.power(eta, 2)
            + 0.4330636324653828 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_PeakAmp_33"] = (
        -0.003288482386411718
        * delta
        * (1 - 8.612308762619447 * eta)
        * eta
        * jnp.power(dchi, 2)
        + delta
        * eta
        * (
            0.5684405079702229
            - 0.00028819674607128055 * eta
            + 2.777740140752971 * jnp.power(eta, 2)
            - 2.3599556709823535 * jnp.power(eta, 3)
        )
        + 0.03887129318550153 * dchi * (1 + 42.30525422235957 * eta) * jnp.power(eta, 3)
        - 0.2051295687108511
        * dchi
        * (1 - 4.34985595987507 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.0652759726861487 * delta * eta
            + 0.25561789058890033 * delta * jnp.power(eta, 2)
            - 1.3134311480695775 * delta * jnp.power(eta, 3)
        )
        + (
            0.04814607684462918 * delta * eta
            - 0.3140983091545102 * delta * jnp.power(eta, 2)
            + 1.1976699463228568 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.03619614547561679 * delta * eta
            - 0.5532673160072701 * delta * jnp.power(eta, 2)
            + 2.4943333040591695 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Amp_C3_33"] = (
        -0.28666660414434536
        + 0.5669087275249756 * eta
        + S
        * (
            -0.22961653919716726
            + 0.7755862716197967 * eta
            - 0.03726170050389395 * jnp.power(eta, 2)
        )
        - 0.2969983864658452 * jnp.power(eta, 2)
        + (
            -0.2177519810696989
            + 1.5186886188134678 * eta
            - 2.1091591639362255 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            0.018605290436426794
            + 0.8121676169377119 * eta
            - 3.309654335397225 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP1_44"] = (
        S
        * (
            0.00929146984958081 * eta
            - 0.058559157503356614 * jnp.power(eta, 2)
            + 0.09641520260278541 * jnp.power(eta, 3)
        )
        - 0.06256463263004813
        * dchi
        * delta
        * (1 - 4.724937783266512 * eta)
        * jnp.power(eta, 3)
        - 0.01735529698327505
        * dchi
        * delta
        * (1 - 3.514044834242014 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.000842117844243168 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + eta
        * (
            0.0799508735514674
            - 2.266747175041431 * eta
            + 49.99562376971802 * jnp.power(eta, 2)
            - 699.4551506778732 * jnp.power(eta, 3)
            + 6096.872857701541 * jnp.power(eta, 4)
            - 33243.794194712485 * jnp.power(eta, 5)
            + 110236.14177804616 * jnp.power(eta, 6)
            - 203224.30569500144 * jnp.power(eta, 7)
            + 159685.76954854574 * jnp.power(eta, 8)
        )
        + (
            0.0036836428878512274 * eta
            - 0.032181253212945134 * jnp.power(eta, 2)
            + 0.06990383731270383 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.005305905044786687 * eta
            - 0.05454070105726642 * jnp.power(eta, 2)
            + 0.1328930616146293 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP2_44"] = (
        S
        * (
            0.032054779889996984 * eta
            - 0.1824264213397133 * jnp.power(eta, 2)
            + 0.2662860950846518 * jnp.power(eta, 3)
        )
        - 0.09838860200524911
        * dchi
        * delta
        * (1 - 4.413878576399552 * eta)
        * jnp.power(eta, 3)
        - 0.07541756416690493
        * dchi
        * delta
        * (1 - 4.896726739338081 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.00024755181872440586 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + eta
        * (
            0.10752001314377323
            - 1.3996074805076077 * eta
            + 17.290345408924 * jnp.power(eta, 2)
            - 146.28994121129182 * jnp.power(eta, 3)
            + 710.8477248404537 * jnp.power(eta, 4)
            - 1819.0962884465648 * jnp.power(eta, 5)
            + 1897.1460245953783 * jnp.power(eta, 6)
        )
        + (
            0.020209039503607196 * eta
            - 0.1635522752682757 * jnp.power(eta, 2)
            + 0.3379077937523624 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.016250056498330504 * eta
            - 0.1599454341389429 * jnp.power(eta, 2)
            + 0.38060091765599724 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP3_44"] = (
        S
        * (
            0.0375923390273927 * eta
            - 0.19675674044979322 * jnp.power(eta, 2)
            + 0.2524073874950236 * jnp.power(eta, 3)
        )
        - 0.10103910572578918
        * dchi
        * delta
        * (1 - 4.5113969567894685 * eta)
        * jnp.power(eta, 3)
        - 0.075429355757026
        * dchi
        * delta
        * (1 - 5.38318094443173 * eta)
        * S
        * jnp.power(eta, 3)
        - 0.0001543369511547082 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + eta
        * (
            0.11684543226973083
            - 1.1708344201904572 * eta
            + 11.160637047449095 * jnp.power(eta, 2)
            - 76.70545398788732 * jnp.power(eta, 3)
            + 299.88284206545273 * jnp.power(eta, 4)
            - 611.534557826681 * jnp.power(eta, 5)
            + 503.71541521565484 * jnp.power(eta, 6)
        )
        + (
            0.025070408583957437 * eta
            - 0.18759667520550588 * jnp.power(eta, 2)
            + 0.36148626759006963 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.02203846168885738 * eta
            - 0.20949146417573655 * jnp.power(eta, 2)
            + 0.4885519836075034 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Merger_Amp_CP1_44"] = (
        S
        * (
            0.058706429585806096 * eta
            - 0.29646515787762634 * jnp.power(eta, 2)
            + 0.3413797381980054 * jnp.power(eta, 3)
        )
        + 0.002277607502351356
        * dchi
        * delta
        * (1 + 205.78624315268536 * eta)
        * jnp.power(eta, 3)
        + 1.4842139797582348e-6
        * dchi
        * delta
        * (1 + 109390.82655130373 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.0034168773116717826 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + eta
        * (
            0.1883789790445804
            - 1.3375818588191817 * eta
            + 14.678591980033465 * jnp.power(eta, 2)
            - 137.42245036191002 * jnp.power(eta, 3)
            + 729.2611054727125 * jnp.power(eta, 4)
            - 2042.3910621805978 * jnp.power(eta, 5)
            + 2339.4396482545535 * jnp.power(eta, 6)
        )
        + (
            0.03693734108438023 * eta
            - 0.2701634652682831 * jnp.power(eta, 2)
            + 0.5277025188909843 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.01916289858506809 * eta
            - 0.16587706623865955 * jnp.power(eta, 2)
            + 0.3873216009596506 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_PeakAmp_44"] = (
        0.697452842995687
        * dchi
        * delta
        * (1 - 1.5644622288381207 * eta)
        * jnp.power(eta, 3)
        + 0.7491313799476855
        * dchi
        * delta
        * (1 - 5.51514415207437 * eta)
        * S
        * jnp.power(eta, 3)
        + 0.03263766742842678 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            0.08013462545147897 * eta
            - 0.707339986581501 * jnp.power(eta, 2)
            + 1.4945536281037473 * jnp.power(eta, 3)
        )
        + eta
        * (
            0.27614097883794725
            - 0.403452380875202 * eta
            - 15.80475783619391 * jnp.power(eta, 2)
            + 227.28867728765587 * jnp.power(eta, 3)
            - 1523.2444219539561 * jnp.power(eta, 4)
            + 4722.659771036674 * jnp.power(eta, 5)
            - 5388.149395981192 * jnp.power(eta, 6)
        )
        + (
            0.0484478773571511 * eta
            - 0.5421173150365266 * jnp.power(eta, 2)
            + 1.5486181139304755 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.019255034163450358 * eta
            - 0.18207194531823234 * jnp.power(eta, 2)
            + 0.4812433162713078 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Amp_C3_44"] = (
        -0.1772709159577312
        - 0.3910604290424687 * eta
        + S
        * (
            -0.14215203243769525
            + 0.6136073658063063 * eta
            + 0.11700379912379351 * jnp.power(eta, 2)
        )
        + 5.876832797574524 * jnp.power(eta, 2)
        + (
            -0.15523859963666756
            + 0.5879889924742473 * eta
            - 3.514395471691389 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            -0.0829048220630192
            - 1.965867892839485 * eta
            + 11.364728644855896 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP1_55"] = (
        -0.0019775643769147514 * dchi * (1 - 4.53924184281778 * eta) * jnp.power(eta, 3)
        + 0.0014385048318273654
        * dchi
        * (1 - 3.856415079702643 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.004541994163205327 * delta * eta
            - 0.032008447973076316 * delta * jnp.power(eta, 2)
            + 0.06489393530989815 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.028336688600531488
            - 0.5409697917194701 * eta
            + 5.981455840165866 * jnp.power(eta, 2)
            - 35.58496309663864 * jnp.power(eta, 3)
            + 106.06125067357719 * jnp.power(eta, 4)
            - 124.75528935423806 * jnp.power(eta, 5)
        )
        + (
            0.0014791305997009444 * delta * eta
            - 0.013564901743537111 * delta * jnp.power(eta, 2)
            + 0.032142792215182195 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP2_55"] = (
        -0.00003724952333242274
        * delta
        * (1 - 3.452222430510181 * eta)
        * eta
        * jnp.power(dchi, 2)
        - 0.009011493135309705
        * dchi
        * (1 - 4.802896680414448 * eta)
        * jnp.power(eta, 3)
        - 0.00013127526660987315
        * dchi
        * (1 - 30.606067223270347 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.018145201665260808 * delta * eta
            - 0.10875155354973318 * delta * jnp.power(eta, 2)
            + 0.1967640499342343 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.05099973356701926
            - 0.9909406291652298 * eta
            + 16.002112346656087 * jnp.power(eta, 2)
            - 151.48298211427934 * jnp.power(eta, 3)
            + 798.2800157600177 * jnp.power(eta, 4)
            - 2185.7904303138503 * jnp.power(eta, 5)
            + 2423.2590529527615 * jnp.power(eta, 6)
        )
        + (
            0.007157848585528541 * delta * eta
            - 0.04884915304244923 * delta * jnp.power(eta, 2)
            + 0.09053190435686395 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Inspiral_Amp_CP3_55"] = (
        0.000014325948759589005
        * delta
        * (1 - 0.6402417006774828 * eta)
        * eta
        * jnp.power(dchi, 2)
        - 0.010476190606527443
        * dchi
        * (1 - 5.143499750443247 * eta)
        * jnp.power(eta, 3)
        - 0.0030979537930086267
        * dchi
        * (1 - 5.795657988613435 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.02269381988843006 * delta * eta
            - 0.12335053938879872 * delta * jnp.power(eta, 2)
            + 0.20474150752693748 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.04442248963378816
            - 0.26245173832295265 * eta
            + 0.03870693239289467 * jnp.power(eta, 2)
            + 20.204218440428264 * jnp.power(eta, 3)
            - 181.23307091031228 * jnp.power(eta, 4)
            + 648.6147508591258 * jnp.power(eta, 5)
            - 849.2143555875599 * jnp.power(eta, 6)
        )
        + (
            0.00913213583658649 * delta * eta
            - 0.04919000144868387 * delta * jnp.power(eta, 2)
            + 0.065986695477459 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_Merger_Amp_CP1_55"] = (
        -0.00031681692142477853
        * delta
        * (1 - 6.678258498810139 * eta)
        * eta
        * jnp.power(dchi, 2)
        + 0.04782012096371179
        * dchi
        * (1 - 2.4136112323522805 * eta)
        * jnp.power(eta, 3)
        + 0.017136472806300446
        * dchi
        * (1 - 2.278716802319452 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.035540705171105316 * delta * eta
            - 0.13367790689815431 * delta * jnp.power(eta, 2)
            + 0.1379312625561125 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.09741180071981874
            - 1.125020791484383 * eta
            + 19.71417205433866 * jnp.power(eta, 2)
            - 206.5659386239305 * jnp.power(eta, 3)
            + 1187.7241527779179 * jnp.power(eta, 4)
            - 3511.371401268973 * jnp.power(eta, 5)
            + 4163.584096800788 * jnp.power(eta, 6)
        )
        + (
            0.018044647105118435 * delta * eta
            - 0.09834030851538031 * delta * jnp.power(eta, 2)
            + 0.18292980258680058 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_PeakAmp_55"] = (
        0.29446699883224503 * dchi * (1 - 2.8126528389736913 * eta) * jnp.power(eta, 3)
        + 0.13166834017031467
        * dchi
        * (1 - 3.6911791457138365 * eta)
        * S
        * jnp.power(eta, 3)
        + S
        * (
            0.04335058078657487 * delta * eta
            - 0.21976500003781027 * delta * jnp.power(eta, 2)
            + 0.1427254606254177 * delta * jnp.power(eta, 3)
        )
        + delta
        * eta
        * (
            0.25471102988397937
            - 6.119431622115874 * eta
            + 125.28192989146497 * jnp.power(eta, 2)
            - 1339.55067240476 * jnp.power(eta, 3)
            + 7641.25542069701 * jnp.power(eta, 4)
            - 22186.340091384493 * jnp.power(eta, 5)
            + 25846.606598287333 * jnp.power(eta, 6)
        )
        + (
            0.029015863810076155 * delta * eta
            - 0.4063151087943421 * delta * jnp.power(eta, 2)
            + 1.419210840554402 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.01147033599820311 * delta * eta
            - 0.28735230830842273 * delta * jnp.power(eta, 2)
            + 1.1999844084222553 * delta * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_RD_Amp_C3_55"] = (
        0.01889156394866289
        - 7.843569775936414 * eta
        + S
        * (
            -0.11458748447064408
            - 0.3369320850812222 * eta
            - 0.022525692986479693 * jnp.power(eta, 2)
        )
        + 73.19838355427139 * jnp.power(eta, 2)
        - 170.9024182786024 * jnp.power(eta, 3)
        + 76.38168535871085 * dchi * (1 - 3.8805106289918205 * eta) * jnp.power(eta, 3)
        + 42.628290542501134
        * dchi
        * (1 - 4.260931557685223 * eta)
        * S
        * jnp.power(eta, 3)
        + (
            -0.18370843418415694
            + 4.918601029356566 * eta
            - 17.29835518657168 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
    )
    fits["IMRPhenomT_tshift_21"] = (
        11.67621653653603
        - 73.94592135375544 * eta
        + 617.5327332811615 * jnp.power(eta, 2)
        + S
        * (
            0.2309485101131543
            - 57.0459017581492 * eta
            + 222.97200099809325 * jnp.power(eta, 2)
        )
        - 2819.458362260437 * jnp.power(eta, 3)
        - 681.9002621172333 * dchi * (1 - 3.989581262513545 * eta) * jnp.power(eta, 3)
        - 1440.639932639621
        * dchi
        * (1 - 4.206805719889809 * eta)
        * S
        * jnp.power(eta, 3)
        + 42.39667266040204 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + 4546.903391979042 * jnp.power(eta, 4)
        + (
            2.2730487886808395
            - 47.65323000340801 * eta
            + 57.297549898351896 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            -1.7237973456372406
            + 21.732949566815307 * eta
            - 187.71334824449366 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_tshift_33"] = (
        6.047225180659371
        - 63.50001473845436 * eta
        + S
        * (
            1.26487072884024
            + 9.789577125790505 * eta
            - 18.51669370705306 * jnp.power(eta, 2)
        )
        + 451.1074541600744 * jnp.power(eta, 2)
        - 893.7051616506715 * jnp.power(eta, 3)
        + (
            3.816104939071836
            - 8.676597277291323 * eta
            - 5.808122950219083 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            2.1374074060226045
            - 1.2219912746034096 * eta
            - 31.342471666791727 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_tshift_44"] = (
        S
        * (
            -5.203270014829841
            + 181.27080583258746 * eta
            - 1529.1896864534942 * jnp.power(eta, 2)
            + 3705.463809339287 * jnp.power(eta, 3)
        )
        + jnp.power(1 - 2.812531081541394 * eta, -1)
        * (
            6.6472023470033585
            - 98.64869153538237 * eta
            + 1148.4724313577744 * jnp.power(eta, 2)
            - 6720.146266369297 * jnp.power(eta, 3)
            + 13400.05768313269 * jnp.power(eta, 4)
        )
        + (
            6.9133369343740565
            - 15.898281197030528 * eta
            - 364.6027054334757 * jnp.power(eta, 2)
            + 1362.455178365237 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            23.15294108414908
            - 333.2730725495644 * eta
            + 1647.4278543557452 * jnp.power(eta, 2)
            - 2702.213569022611 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
    )
    fits["IMRPhenomT_tshift_55"] = (
        -0.3189869259194407
        + 153.08687719603935 * eta
        - 1376.0895569730135 * jnp.power(eta, 2)
        + S
        * (
            10.62118364975699
            - 128.69299551679973 * eta
            + 401.7008544741773 * jnp.power(eta, 2)
        )
        + 3511.1779067574766 * jnp.power(eta, 3)
        + (
            10.036441054322665
            - 75.42317272972994 * eta
            + 180.54334490779055 * jnp.power(eta, 2)
        )
        * jnp.power(S, 2)
        + (
            6.658297274982426
            - 35.88874981710895 * eta
            + 33.86225466072782 * jnp.power(eta, 2)
        )
        * jnp.power(S, 3)
    )
    # Missing from the original mega-function (added for the IMRPhenomT (2,2)-mode
    # coefficient-solve, see jaxpe/gw/cbc_models/phenomt.py): the (2,2) inspiral
    # frequency collocation points + TaylorT3 t0, the (2,2) merger frequency collocation
    # point, and the PhenomX final mass/spin fits IMRPhenomTHM uses for the remnant.
    # Mechanically transcribed from LALSimIMRPhenomTHM_fits.c / LALSimIMRPhenomXUtilities.c,
    # matching this file's existing plain-transcription style (no shared-subexpression
    # hand-optimization beyond what's already common to the eta/S/dchi/delta powers above).
    fits["IMRPhenomT_Inspiral_TaylorT3_t0"] = jnp.power(eta, -1) * (
        (-20.74399646637014 - 106.27711276502542 * eta)
        * jnp.power(1 + 0.6516016033332481 * eta, -1)
        + 0.0012450290074562259
        * dchi
        * delta
        * (1 - 4.701633367918768e6 * eta)
        * jnp.power(eta, 2)
        - 111.5049997379579
        * dchi
        * delta
        * (1 + 19.95458485773613 * eta)
        * S
        * jnp.power(eta, 2)
        + 1204.6829118499857
        * (1 - 4.025474056585855 * eta)
        * jnp.power(dchi, 2)
        * jnp.power(eta, 3)
        + S
        * (
            338.7318821277009
            - 1553.5891860091408 * eta
            + 19614.263378999745 * jnp.power(eta, 2)
            - 156449.78737303324 * jnp.power(eta, 3)
            + 577363.3090369126 * jnp.power(eta, 4)
            - 802867.433363341 * jnp.power(eta, 5)
        )
        + (
            -55.75053935847546
            - 290.36341163610575 * eta
            + 7873.7667183299345 * jnp.power(eta, 2)
            - 43585.59040070178 * jnp.power(eta, 3)
            + 87229.84668746481 * jnp.power(eta, 4)
            - 32469.263449695136 * jnp.power(eta, 5)
        )
        * jnp.power(S, 2)
        + (
            -102.8269343111326
            + 5121.845705262981 * eta
            - 93026.46878769135 * jnp.power(eta, 2)
            + 650989.6793529999 * jnp.power(eta, 3)
            - 1.8846061037110784e6 * jnp.power(eta, 4)
            + 1.861602620702142e6 * jnp.power(eta, 5)
        )
        * jnp.power(S, 3)
        + (
            -7.294950933078567
            + 314.24955197427136 * eta
            - 3751.8509582195657 * jnp.power(eta, 2)
            + 21205.339564205595 * jnp.power(eta, 3)
            - 46448.94771114493 * jnp.power(eta, 4)
            + 20310.512558558552 * jnp.power(eta, 5)
        )
        * jnp.power(S, 4)
        + (
            97.22312282683716
            - 4556.60375328623 * eta
            + 76308.73046927384 * jnp.power(eta, 2)
            - 468784.4188333802 * jnp.power(eta, 3)
            + 998692.0246600509 * jnp.power(eta, 4)
            - 322905.9042578296 * jnp.power(eta, 5)
        )
        * jnp.power(S, 5)
    )
    fits["IMRPhenomT_Inspiral_Freq_CP1_22"] = (
        -0.014968864336704284
        * dchi
        * delta
        * (1 - 1.942061808318584 * eta)
        * jnp.power(eta, 2)
        + 0.0017312772309375462
        * dchi
        * delta
        * (1 - 0.07106994121956058 * eta)
        * S
        * jnp.power(eta, 2)
        + S
        * (
            0.0019208448318368731
            - 0.0013579968243452476 * eta
            - 0.0033501404728414627 * jnp.power(eta, 2)
            + 0.008914420175326192 * jnp.power(eta, 3)
        )
        + 6.687615165457298e-6 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + (
            0.02104073275966069
            + 717.1534194224539 * eta
            + 85.37320237350282 * jnp.power(eta, 2)
            + 12.789214868358362 * jnp.power(eta, 3)
            - 16.00243777208413 * jnp.power(eta, 4)
        )
        * jnp.power(1 + 32934.586638893634 * eta, -1)
        + (
            -8.306810248117731e-6
            + 0.00009918593182087119 * eta
            - 0.003805916669791129 * jnp.power(eta, 2)
            + 0.009854209286892323 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            -5.578836442449699e-6
            - 0.0030378960591856616 * eta
            + 0.03746366675135751 * jnp.power(eta, 2)
            - 0.10298471015315146 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
        + (
            0.00004425141111368952
            - 0.0008702073302258368 * eta
            + 0.006538604805919268 * jnp.power(eta, 2)
            - 0.01578597166324495 * jnp.power(eta, 3)
        )
        * jnp.power(S, 4)
        + (
            -0.000019469656288570753
            + 0.002969863931498354 * eta
            - 0.03643271052162611 * jnp.power(eta, 2)
            + 0.09959495981802587 * jnp.power(eta, 3)
        )
        * jnp.power(S, 5)
        + (
            -0.000042037164406446896
            + 0.0007336074135429041 * eta
            - 0.005603356997202016 * jnp.power(eta, 2)
            + 0.013439843000090702 * jnp.power(eta, 3)
        )
        * jnp.power(S, 6)
    )
    fits["IMRPhenomT_Inspiral_Freq_CP2_22"] = (
        -0.04486391236129559
        * dchi
        * delta
        * (1 - 1.8997912248414794 * eta)
        * jnp.power(eta, 2)
        - 0.003531802135161727
        * dchi
        * delta
        * (1 - 8.001211450141325 * eta)
        * S
        * jnp.power(eta, 2)
        + S
        * (
            0.0061664395419698285
            - 0.0040934633081508905 * eta
            - 0.009180337242551828 * jnp.power(eta, 2)
            + 0.020338583755834694 * jnp.power(eta, 3)
        )
        + 0.00006524644306613066 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + jnp.power(1 - 3.2125452791404148 * eta, -1)
        * (
            0.03711511661217631
            - 0.10663782888636487 * eta
            - 0.09963406984414182 * jnp.power(eta, 2)
            + 0.6597367702009397 * jnp.power(eta, 3)
            - 2.777344875144891 * jnp.power(eta, 4)
            + 4.220674345359693 * jnp.power(eta, 5)
        )
        + (
            0.00044302547647888445
            + 0.000424246501303979 * eta
            - 0.01394093576260671 * jnp.power(eta, 2)
            + 0.02634851560709597 * jnp.power(eta, 3)
        )
        * jnp.power(S, 2)
        + (
            0.00011582043047950321
            - 0.008282652950117982 * eta
            + 0.08965067576998058 * jnp.power(eta, 2)
            - 0.23963885130463913 * jnp.power(eta, 3)
        )
        * jnp.power(S, 3)
        + (
            0.0006123158975881322
            - 0.007809160444435783 * eta
            + 0.028517174579539676 * jnp.power(eta, 2)
            - 0.03717957419042746 * jnp.power(eta, 3)
        )
        * jnp.power(S, 4)
        + (
            -0.0000885530893214531
            + 0.005939789043536808 * eta
            - 0.07106551435109858 * jnp.power(eta, 2)
            + 0.1891131957235774 * jnp.power(eta, 3)
        )
        * jnp.power(S, 5)
        + (
            -0.0005110853374341054
            + 0.0038762476596420855 * eta
            + 0.005094077179675256 * jnp.power(eta, 2)
            - 0.047971766995287136 * jnp.power(eta, 3)
        )
        * jnp.power(S, 6)
    )
    fits["IMRPhenomT_Inspiral_Freq_CP3_22"] = (
        -0.10196878573773932
        * dchi
        * delta
        * (1 - 1.8918584778973513 * eta)
        * jnp.power(eta, 2)
        - 0.018820536453940443
        * dchi
        * delta
        * (1 - 3.7307154599131183 * eta)
        * S
        * jnp.power(eta, 2)
        - 0.00013162098437956188 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            0.0145572994468378
            - 0.0017482433991394227 * eta
            - 0.10299007619034371 * jnp.power(eta, 2)
            + 0.4581039376357615 * jnp.power(eta, 3)
            - 0.7123678787549022 * jnp.power(eta, 4)
        )
        + (
            0.05489007025458171
            + 5.852073438961151 * eta
            + 2.74597705533403 * jnp.power(eta, 2)
            + 4.834336623113389 * jnp.power(eta, 3)
            - 26.931994454691022 * jnp.power(eta, 4)
            + 57.67035368809743 * jnp.power(eta, 5)
        )
        * jnp.power(1 + 105.52132834236778 * eta, -1)
        + (
            0.003001211395915229
            + 0.0017929418998452987 * eta
            - 0.13776590125456148 * jnp.power(eta, 2)
            + 0.7471133710854526 * jnp.power(eta, 3)
            - 1.3620323111858437 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            0.001143282743686261
            - 0.05793457776296727 * eta
            + 0.7841331051705482 * jnp.power(eta, 2)
            - 3.4936244160305323 * jnp.power(eta, 3)
            + 4.802357041496856 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
        + (
            0.0009168588840889624
            - 0.03261437094899735 * eta
            + 0.3472881896838799 * jnp.power(eta, 2)
            - 1.3634383958859384 * jnp.power(eta, 3)
            + 1.7313939586675267 * jnp.power(eta, 4)
        )
        * jnp.power(S, 4)
        + (
            -0.0002794014744432316
            + 0.055911057147527664 * eta
            - 0.8686311380514122 * jnp.power(eta, 2)
            + 4.096191294930781 * jnp.power(eta, 3)
            - 6.009676060669872 * jnp.power(eta, 4)
        )
        * jnp.power(S, 5)
        + (
            -0.0005046018052528331
            + 0.029804593053788925 * eta
            - 0.3792653361049425 * jnp.power(eta, 2)
            + 1.6366976231421981 * jnp.power(eta, 3)
            - 2.26904099961476 * jnp.power(eta, 4)
        )
        * jnp.power(S, 6)
    )
    fits["IMRPhenomT_Inspiral_Freq_CP4_22"] = (
        -0.1831889759662071
        * dchi
        * delta
        * (1 - 1.8484261527766557 * eta)
        * jnp.power(eta, 2)
        - 0.07586202965525136
        * dchi
        * delta
        * (1 - 3.2918162656371983 * eta)
        * S
        * jnp.power(eta, 2)
        + 0.0019259052728265817 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            0.02685637375751212
            + 0.013341664908359861 * eta
            - 0.3057217933283597 * jnp.power(eta, 2)
            + 1.395763446325911 * jnp.power(eta, 3)
            - 2.2559396974665376 * jnp.power(eta, 4)
        )
        + (
            0.0725639467287476
            + 12.39400068457852 * eta
            + 12.907450928972402 * jnp.power(eta, 2)
            - 7.422660061864399 * jnp.power(eta, 3)
            + 66.32985901506036 * jnp.power(eta, 4)
            - 117.85875779454518 * jnp.power(eta, 5)
        )
        * jnp.power(1 + 168.63492460136445 * eta, -1)
        + (
            0.0087781653701194
            + 0.006944161553839352 * eta
            - 0.3301149078235105 * jnp.power(eta, 2)
            + 1.6835714783903248 * jnp.power(eta, 3)
            - 2.950404929598742 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            0.0037229746496019625
            - 0.17155338099487646 * eta
            + 2.5881802140836774 * jnp.power(eta, 2)
            - 13.14710199375518 * jnp.power(eta, 3)
            + 21.366803256010915 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
        + (
            0.00278507305662002
            - 0.12475855143364532 * eta
            + 1.8640209516178643 * jnp.power(eta, 2)
            - 10.117078727717564 * jnp.power(eta, 3)
            + 17.94244821676711 * jnp.power(eta, 4)
        )
        * jnp.power(S, 4)
        + (
            0.0010273954584773936
            + 0.1713357629442166 * eta
            - 3.017249223460983 * jnp.power(eta, 2)
            + 15.855096360798678 * jnp.power(eta, 3)
            - 26.444621592311933 * jnp.power(eta, 4)
        )
        * jnp.power(S, 5)
        + (
            -0.00012207946532225968
            + 0.11709700788855186 * eta
            - 2.0950821618097026 * jnp.power(eta, 2)
            + 11.925324501640054 * jnp.power(eta, 3)
            - 21.683978511818076 * jnp.power(eta, 4)
        )
        * jnp.power(S, 6)
    )
    fits["IMRPhenomT_Inspiral_Freq_CP5_22"] = (
        -0.2508206617297265
        * dchi
        * delta
        * (1 - 1.861010982421798 * eta)
        * jnp.power(eta, 2)
        - 0.1392163711259171
        * dchi
        * delta
        * (1 - 3.2669366465555796 * eta)
        * S
        * jnp.power(eta, 2)
        + 0.0023126403170013045 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            0.036750064163293766
            + 0.036904343404333906 * eta
            - 0.5238739410356437 * jnp.power(eta, 2)
            + 2.3292117112945223 * jnp.power(eta, 3)
            - 3.654184701923543 * jnp.power(eta, 4)
        )
        + (
            0.08373610487663233
            + 6.301736487754372 * eta
            + 9.03911386193751 * jnp.power(eta, 2)
            + 4.91153188278086 * jnp.power(eta, 3)
        )
        * jnp.power(1 + 72.64820846804257 * eta, -1)
        + (
            0.014963449678540705
            + 0.008354571522567225 * eta
            - 0.41723078020683 * jnp.power(eta, 2)
            + 2.2007932082378785 * jnp.power(eta, 3)
            - 4.245354787320365 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            0.005706180633326235
            - 0.15748500622007494 * eta
            + 2.3477109912232845 * jnp.power(eta, 2)
            - 11.413877195221694 * jnp.power(eta, 3)
            + 17.033120593116756 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
        + (
            0.003890296981717687
            - 0.15985471334551038 * eta
            + 2.560312006077997 * jnp.power(eta, 2)
            - 14.400920672743332 * jnp.power(eta, 3)
            + 26.10406142567958 * jnp.power(eta, 4)
        )
        * jnp.power(S, 4)
        + (
            0.005305988847210204
            + 0.10869207132210629 * eta
            - 2.4201307115268875 * jnp.power(eta, 2)
            + 12.544899744864924 * jnp.power(eta, 3)
            - 19.550600837316903 * jnp.power(eta, 4)
        )
        * jnp.power(S, 5)
        + (
            0.002917248769788225
            + 0.11851143848720952 * eta
            - 2.6640023622893416 * jnp.power(eta, 2)
            + 15.993378498844761 * jnp.power(eta, 3)
            - 29.752144941054446 * jnp.power(eta, 4)
        )
        * jnp.power(S, 6)
    )
    fits["IMRPhenomT_Merger_Freq_CP1_22"] = (
        -0.3926039690467202
        * dchi
        * delta
        * (1 - 2.359180951434749 * eta)
        * jnp.power(eta, 2)
        - 0.28551098014898896
        * dchi
        * delta
        * (1 - 3.414696100901444 * eta)
        * S
        * jnp.power(eta, 2)
        + 0.003414004344822246 * jnp.power(dchi, 2) * jnp.power(eta, 3)
        + S
        * (
            0.05697014130854102
            + 0.07170430925984912 * eta
            - 0.9606499306623374 * jnp.power(eta, 2)
            + 5.440955307244598 * jnp.power(eta, 3)
            - 10.594319036394571 * jnp.power(eta, 4)
        )
        + (
            0.10030959768350425
            + 44.56725135920024 * eta
            + 163.96290948585087 * jnp.power(eta, 2)
            - 143.05635831020462 * jnp.power(eta, 3)
            + 393.8084861740473 * jnp.power(eta, 4)
        )
        * jnp.power(1 + 436.6494065618 * eta, -1)
        + (
            0.021213606590798472
            + 0.2148355967310081 * eta
            - 2.7747405367196265 * jnp.power(eta, 2)
            + 13.771088220299802 * jnp.power(eta, 3)
            - 25.128755397215368 * jnp.power(eta, 4)
        )
        * jnp.power(S, 2)
        + (
            -0.003645992092251503
            + 0.2137524962844931 * eta
            - 0.644979226062801 * jnp.power(eta, 2)
            - 1.7314849842209137 * jnp.power(eta, 3)
            + 5.573297392347478 * jnp.power(eta, 4)
        )
        * jnp.power(S, 3)
        + (
            0.029352214609533665
            - 0.6020287633594307 * eta
            + 7.014738679280164 * jnp.power(eta, 2)
            - 36.027159248248296 * jnp.power(eta, 3)
            + 63.42605850359639 * jnp.power(eta, 4)
        )
        * jnp.power(S, 4)
        + (
            0.0356519646654399
            - 0.5569780178251297 * eta
            + 4.017784725334053 * jnp.power(eta, 2)
            - 15.05881246593488 * jnp.power(eta, 3)
            + 22.94821359434365 * jnp.power(eta, 4)
        )
        * jnp.power(S, 5)
    )

    eta2, eta3, eta4 = jnp.power(eta, 2), jnp.power(eta, 3), jnp.power(eta, 4)
    S2, S3 = jnp.power(S, 2), jnp.power(S, 3)
    dchi2 = jnp.power(dchi, 2)
    _no_spin_mass = (
        0.057190958417936644 * eta
        + 0.5609904135313374 * eta2
        - 0.84667563764404 * eta3
        + 3.145145224278187 * eta4
    )
    _eq_spin_mass = (
        _no_spin_mass
        * (
            1
            + (
                -0.13084389181783257
                - 1.1387311580238488 * eta
                + 5.49074464410971 * eta2
            )
            * S
            + (-0.17762802148331427 + 2.176667900182948 * eta2) * S2
            + (
                -0.6320191645391563
                + 4.952698546796005 * eta
                - 10.023747993978121 * eta2
            )
            * S3
        )
        / (
            1
            + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2)
            * S
        )
        - _no_spin_mass
    )
    _uneq_spin_mass = (
        -0.09803730445895877 * dchi * delta * (1 - 3.2283713377939134 * eta) * eta2
        + 0.01118530335431078 * dchi2 * eta3
        - 0.01978238971523653 * dchi * delta * (1 - 4.91667749015812 * eta) * eta * S
    )
    fits["IMRPhenomX_FinalMass2017"] = 1.0 - (
        _no_spin_mass + _eq_spin_mass + _uneq_spin_mass
    )

    _m1 = 0.5 * (1.0 + delta)
    _m2 = 0.5 * (1.0 - delta)
    _no_spin_af = (
        3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3
    ) / (1 + 7.2388440419467335 * eta)
    _eq_spin_af = (_m1 * _m1 + _m2 * _m2) * S + (
        (
            -0.8561951310209386 * eta
            - 0.09939065676370885 * eta2
            + 1.668810429851045 * eta3
        )
        * S
        + (
            0.5881660363307388 * eta
            - 2.149269067519131 * eta2
            + 3.4768263932898678 * eta3
        )
        * S2
        + (
            0.142443244743048 * eta
            - 0.9598353840147513 * eta2
            + 1.9595643107593743 * eta3
        )
        * S3
    ) / (
        1
        + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3)
        * S
    )
    _uneq_spin_af = (
        0.3223660562764661 * dchi * delta * (1 + 9.332575956437443 * eta) * eta2
        - 0.059808322561702126 * dchi2 * eta3
        + 2.3170397514509933 * dchi * delta * (1 - 3.2624649875884852 * eta) * eta3 * S
    )
    fits["IMRPhenomX_FinalSpin2017"] = _no_spin_af + _eq_spin_af + _uneq_spin_af

    return fits


@jax.jit
def evaluate_QNMfit_fring22(a):
    """(2,2) ringdown frequency fit (dimensionless, Mf), as a function of final spin."""
    a2, a3, a4, a5, a6, a7 = (jnp.power(a, n) for n in (2, 3, 4, 5, 6, 7))
    num = (
        0.05947169566573468
        - 0.14989771215394762 * a
        + 0.09535606290986028 * a2
        + 0.02260924869042963 * a3
        - 0.02501704155363241 * a4
        - 0.005852438240997211 * a5
        + 0.0027489038393367993 * a6
        + 0.0005821983163192694 * a7
    )
    den = (
        1
        - 2.8570126619966296 * a
        + 2.373335413978394 * a2
        - 0.6036964688511505 * a4
        + 0.0873798215084077 * a6
    )
    return num / den


@jax.jit
def evaluate_QNMfit_fdamp22(a):
    """(2,2) damping frequency fit (dimensionless, Mf), as a function of final spin."""
    a2, a3, a4, a5, a6 = (jnp.power(a, n) for n in (2, 3, 4, 5, 6))
    num = (
        0.014158792290965177
        - 0.036989395871554566 * a
        + 0.026822526296575368 * a2
        + 0.0008490933750566702 * a3
        - 0.004843996907020524 * a4
        - 0.00014745235759327472 * a5
        + 0.0001504546201236794 * a6
    )
    den = (
        1
        - 2.5900842798681376 * a
        + 1.8952576220623967 * a2
        - 0.31416610693042507 * a4
        + 0.009002719412204133 * a6
    )
    return num / den


@jax.jit
def evaluate_QNMfit_fdamp22n2(a):
    """(2,2) n=2-overtone damping frequency fit (dimensionless, Mf), function of final spin."""
    a2, a3, a4, a5, a6, a7, a8 = (jnp.power(a, n) for n in (2, 3, 4, 5, 6, 7, 8))
    return 0.043611742588188715 + (
        -0.004016191313442792 * a
        - 0.0027646155943395426 * a2
        + 0.001141927763953028 * a3
        + 0.007938320030300492 * a4
        - 0.0008263166671238823 * a5
        - 0.014025760257115768 * a6
        + 0.001792158578158245 * a7
        + 0.008824138122361842 * a8
    ) / (2.0 - 1.9477781396815619 * a)


@jax.jit
def evaluate_QNMfit_fring21(a):
    x2, x3, x4, x5 = (jnp.power(a, n) for n in (2, 3, 4, 5))
    return (
        0.059471695665734674
        - 0.07585416297991414 * a
        + 0.021967909664591865 * x2
        - 0.0018964744613388146 * x3
        + 0.001164879406179587 * x4
        - 0.0003387374454044957 * x5
    ) / (1 - 1.4437415542456158 * a + 0.49246920313191234 * x2)


@jax.jit
def evaluate_QNMfit_fring33(a):
    x2, x3, x4, x5, x6 = (jnp.power(a, n) for n in (2, 3, 4, 5, 6))
    return (
        0.09540436245212061
        - 0.22799517865876945 * a
        + 0.13402916709362475 * x2
        + 0.03343753057911253 * x3
        - 0.030848060170259615 * x4
        - 0.006756504382964637 * x5
        + 0.0027301732074159835 * x6
    ) / (
        1
        - 2.7265947806178334 * a
        + 2.144070539525238 * x2
        - 0.4706873667569393 * x4
        + 0.05321818246993958 * x6
    )


@jax.jit
def evaluate_QNMfit_fring44(a):
    x2, x3, x4, x5, x6 = (jnp.power(a, n) for n in (2, 3, 4, 5, 6))
    return (
        0.1287821193485683
        - 0.21224284094693793 * a
        + 0.0710926778043916 * x2
        + 0.015487322972031054 * x3
        - 0.002795401084713644 * x4
        + 0.000045483523029172406 * x5
        + 0.00034775290179000503 * x6
    ) / (
        1 - 1.9931645124693607 * a + 1.0593147376898773 * x2 - 0.06378640753152783 * x4
    )


@jax.jit
def evaluate_QNMfit_fring55(a):
    x = a
    return 0.16110773330909547 + (
        0.056600832610159385 * x
        + 0.030041275213566483 * jnp.power(x, 2)
        - 0.07522309632456432 * jnp.power(x, 3)
        - 0.036341969761668556 * jnp.power(x, 4)
        + 0.015617599737487714 * jnp.power(x, 5)
        + 0.0062588909671250715 * jnp.power(x, 6)
        + 0.004242111725892476 * jnp.power(x, 7)
        + 0.0014913342466081074 * jnp.power(x, 8)
    ) * jnp.power(
        1
        + 0.008923302958356548 * x
        - 1.666395858912649 * jnp.power(x, 2)
        + 0.6697719493836555 * jnp.power(x, 4),
        -1,
    )


@jax.jit
def evaluate_QNMfit_fdamp21(a):
    x2, x3, x4, x5 = (jnp.power(a, n) for n in (2, 3, 4, 5))
    return (
        2.0696914454467294
        - 3.1358071947583093 * a
        + 0.14456081596393977 * x2
        + 1.2194717985037946 * x3
        - 0.2947372598589144 * x4
        + 0.002943057145913646 * x5
    ) / (
        146.1779212636481
        - 219.81790388304876 * a
        + 17.7141194900164 * x2
        + 75.90115083917898 * x3
        - 18.975287709794745 * x4
    )


@jax.jit
def evaluate_QNMfit_fdamp33(a):
    x2, x3, x4, x5 = (jnp.power(a, n) for n in (2, 3, 4, 5))
    return (
        0.014754148319335946
        - 0.03124423610028678 * a
        + 0.017192623913708124 * x2
        + 0.001034954865629645 * x3
        - 0.0015925124814622795 * x4
        - 0.0001414350555699256 * x5
    ) / (1 - 2.0963684630756894 * a + 1.196809702382645 * x2 - 0.09874113387889819 * x4)


@jax.jit
def evaluate_QNMfit_fdamp44(a):
    x2, x3, x4, x5, x6 = (jnp.power(a, n) for n in (2, 3, 4, 5, 6))
    return (
        0.014986847152355699
        - 0.01722587715950451 * a
        - 0.0016734788189065538 * x2
        + 0.0002837322846047305 * x3
        + 0.002510528746148588 * x4
        + 0.00031983835498725354 * x5
        + 0.000812185411753066 * x6
    ) / (
        1
        - 1.1350205970682399 * a
        - 0.0500827971270845 * x2
        + 0.13983808071522857 * x4
        + 0.051876225199833995 * x6
    )


@jax.jit
def evaluate_QNMfit_fdamp55(a):
    x = a
    return 0.015104212245401403 + jnp.power(2 - 1.9485458003209648 * x, -1) * (
        -0.0002946999837678157 * x
        - 0.0024189312940399916 * jnp.power(x, 2)
        + 0.0002099427928656942 * jnp.power(x, 3)
        + 0.00258435043118687 * jnp.power(x, 4)
        - 0.00020630579058983925 * jnp.power(x, 5)
        - 0.004126708789254023 * jnp.power(x, 6)
        + 0.0007950067180727237 * jnp.power(x, 7)
        + 0.0027916616982894588 * jnp.power(x, 8)
    )


@jax.jit
def evaluate_QNMfit_fdamp21n2(a):
    x = a
    return 0.04357957255256736 + jnp.power(2 - 1.9092143068452778 * x, -1) * (
        -0.0019991187832937543 * x
        - 0.00397223929602004 * jnp.power(x, 2)
        + 0.0027170335545048836 * jnp.power(x, 3)
        - 0.003787735584625901 * jnp.power(x, 4)
        + 0.003238742776891051 * jnp.power(x, 5)
        + 0.0014093180629203572 * jnp.power(x, 6)
    )


@jax.jit
def evaluate_QNMfit_fdamp33n2(a):
    x = a
    return 0.04478453069660422 + jnp.power(2 - 1.9490123990107866 * x, -1) * (
        -0.0027276947367212184 * x
        - 0.005325382420460958 * jnp.power(x, 2)
        + 0.0011090264831122598 * jnp.power(x, 3)
        + 0.007374826520017088 * jnp.power(x, 4)
        - 0.000513882756528504 * jnp.power(x, 5)
        - 0.011798583916595289 * jnp.power(x, 6)
        + 0.002064124132395282 * jnp.power(x, 7)
        + 0.007865115260801307 * jnp.power(x, 8)
    )


@jax.jit
def evaluate_QNMfit_fdamp44n2(a):
    x = a
    return 0.04526815749399381 + jnp.power(2 - 1.9488568618006608 * x, -1) * (
        -0.001778614725637923 * x
        - 0.00645234041653255 * jnp.power(x, 2)
        + 0.0008619365083550613 * jnp.power(x, 3)
        + 0.0076173707591557305 * jnp.power(x, 4)
        - 0.0005521040642302851 * jnp.power(x, 5)
        - 0.012109903894557721 * jnp.power(x, 6)
        + 0.0022638317039992374 * jnp.power(x, 7)
        + 0.008166822924109219 * jnp.power(x, 8)
    )


@jax.jit
def evaluate_QNMfit_fdamp55n2(a):
    x = a
    return 0.04550451880252191 + jnp.power(2 - 1.948578997793657 * x, -1) * (
        -0.0012254856066171247 * x
        - 0.007001556265084966 * jnp.power(x, 2)
        + 0.0006934110689396443 * jnp.power(x, 3)
        + 0.007758785424957949 * jnp.power(x, 4)
        - 0.0005928556371123753 * jnp.power(x, 5)
        - 0.012383887808125627 * jnp.power(x, 6)
        + 0.002365118383999583 * jnp.power(x, 7)
        + 0.00838080791280435 * jnp.power(x, 8)
    )
