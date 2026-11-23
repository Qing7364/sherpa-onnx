# 逐样本差异审查

固定同一人工参考与归一化，对所有 600 个配对结果重新核算编辑距离和样本身份。

## A-compatibility

文本/token 不同：0/200。

## B-fp32-batched

文本/token 不同：0/200。

## C-int8-batched

文本/token 不同：1/200。

### BAC009S0908W0389 (zh)

参考：但  喜欢  欧美  音乐  爱吃  西红柿  意大利  面

A：\udc8d

B：但喜欢欧美音乐，爱吃西红柿意大利面。

错误数 16 → 0；新空输出=False，新截断=False，新循环候选=False。

## 官方 INT8 包 → 同源重导出 all INT8

此比较包含发布包与重导出流程/前端差异，不是本次 all/last 优化对照。发布包未提供可核实的原始 checkpoint revision，不能假定内部权重来源完全相同。

| 语言 | 官方错误数/参考长度 | 重导出 all 错误数/参考长度 | 错误率 | 变化样本 |
| --- | --- | --- | --- | --- |
| zh | 44/1461 | 42/1461 | 3.0116% → 2.8747% | 18 |
| en | 76/2166 | 76/2166 | 3.5088% → 3.5088% | 22 |

### 1188-133604-0044

参考：IT WILL BE WELL FOR YOU IF YOU JOIN NOT WITH THOSE WHO INSTEAD OF KITES FLY FALCONS WHO INSTEAD OF OBEYING THE LAST WORDS OF THE GREAT CLOUD SHEPHERD TO FEED HIS SHEEP LIVE THE LIVES HOW MUCH LESS THAN VANITY OF THE WAR WOLF AND THE GIER EAGLE

官方：It will be well for you, if you join not with those who, instead of kites, fly falcons, who, instead of obeying the last words of the great cloud shepherd, to feed his sheep, live the lives. How much less than vanity, of the warwolf and the gear eagle.

重导出 all：It will be well for you if you join not with those who, instead of kites, fly falcons, who, instead of obeying the last words of the great cloud shepherd, to feed his sheep, live the lives. How much less than vanity of the werewolf and the gear eagle.

### 121-123859-0001

参考：O TIS THE FIRST TIS FLATTERY IN MY SEEING AND MY GREAT MIND MOST KINGLY DRINKS IT UP MINE EYE WELL KNOWS WHAT WITH HIS GUST IS GREEING AND TO HIS PALATE DOTH PREPARE THE CUP IF IT BE POISON'D TIS THE LESSER SIN THAT MINE EYE LOVES IT AND DOTH FIRST BEGIN

官方：Oh, tis the first tis flattery in my seeing, and my great mind most kingly drinks it up. Mine eye well knows what with his gust is greeing, and to his palate doth prepare the cup. If it be poisoned, tis the lesser sin, that mine eye loves it, and doth first begin.

重导出 all：Oh, tis the first, tis flattery in my seeing, and my great mind most kingly drinks it up. Mine eye well knows what with his gust is greeing, and to his palate doth prepare the cup, if it be poisoned. Tis the lesser sin, that mine eye loves it, and doth first begin.

### 121-123859-0002

参考：BUT RECKONING TIME WHOSE MILLION'D ACCIDENTS CREEP IN TWIXT VOWS AND CHANGE DECREES OF KINGS TAN SACRED BEAUTY BLUNT THE SHARP'ST INTENTS DIVERT STRONG MINDS TO THE COURSE OF ALTERING THINGS ALAS WHY FEARING OF TIME'S TYRANNY MIGHT I NOT THEN SAY NOW I LOVE YOU BEST WHEN I WAS CERTAIN O'ER INCERTAINTY CROWNING THE PRESENT DOUBTING OF THE REST

官方：But reckoning time, whose million accidents creep in twixt vows, and change decrees of kings, tans sacred beauty blunt the sharpest intents, divert strong minds to the course of altering things. Alas, why fearing of time's tyranny? Might I not then say, "Now I love you best," when I was certain or incertainty, crowning the present, doubting of the rest.

重导出 all：But reckoning time, whose million accidents creep in twixt vows, and change decrees of kings, tan sacred beauty blunt the sharpest intents, divert strong minds to the course of altering things. Alas, why fearing of time's tyranny? Might I not then say, "Now I love you best," when I was certain or in certainty, crowning the present, doubting of the rest.

### 1221-135767-0000

参考：HESTER PRYNNE WENT ONE DAY TO THE MANSION OF GOVERNOR BELLINGHAM WITH A PAIR OF GLOVES WHICH SHE HAD FRINGED AND EMBROIDERED TO HIS ORDER AND WHICH WERE TO BE WORN ON SOME GREAT OCCASION OF STATE FOR THOUGH THE CHANCES OF A POPULAR ELECTION HAD CAUSED THIS FORMER RULER TO DESCEND A STEP OR TWO FROM THE HIGHEST RANK HE STILL HELD AN HONOURABLE AND INFLUENTIAL PLACE AMONG THE COLONIAL MAGISTRACY

官方：Hester Prin went one day to the mansion of Governor Bellingham with a pair of gloves which she had fringed and embroidered to his order, and which were to be worn on some great occasion of state. For though the chances of a popular election had caused this former ruler to descend a step or two from the highest rank, he still held an honorable and influential place among the colonial magistracy.

重导出 all：Hester Prynne went one day to the mansion of Governor Bellingham with a pair of gloves which she had fringed and embroidered to his order, and which were to be worn on some great occasion of state. For though the chances of a popular election had caused this former ruler to descend a step or two from the highest rank, he still held an honorable and influential place among the colonial magistracy.

### 1580-141084-0025

参考：YOU KNOW HIM I THINK SO

官方：You know him, I think so.

重导出 all：You know him. I think so.

### 2094-142345-0000

参考：IT IS A VERY FINE OLD PLACE OF RED BRICK SOFTENED BY A PALE POWDERY LICHEN WHICH HAS DISPERSED ITSELF WITH HAPPY IRREGULARITY SO AS TO BRING THE RED BRICK INTO TERMS OF FRIENDLY COMPANIONSHIP WITH THE LIMESTONE ORNAMENTS SURROUNDING THE THREE GABLES THE WINDOWS AND THE DOOR PLACE

官方：It is a very fine old place of red brick, softened by a pale, powdery lichen, which has dispersed itself with happy irregularity, so as to bring the red brick into terms of friendly companionship with the limestone ornaments surrounding the three gables, the windows, and the doorplace.

重导出 all：It is a very fine old place of red brick, softened by a pale powdery lichen, which has dispersed itself with happy irregularity, so as to bring the red brick into terms of friendly companionship with the limestone ornaments surrounding the three gables, the windows, and the doorplace.

### 2961-961-0002

参考：AND THEREFORE TO YOU I TURN TIMAEUS CITIZEN OF LOCRIS WHO ARE AT ONCE A PHILOSOPHER AND A STATESMAN AND TO YOU CRITIAS WHOM ALL ATHENIANS KNOW TO BE SIMILARLY ACCOMPLISHED AND TO HERMOCRATES WHO IS ALSO FITTED BY NATURE AND EDUCATION TO SHARE IN OUR DISCOURSE

官方：And therefore, to you, I turn to me as citizen of Locris, who are at once a philosopher and a statesman, and to you, Cretius, whom all Athenians know to be similarly accomplished, and to Hermocrates, who is also fitted by nature and education to share in our discourse.

重导出 all：And therefore, to you I turn, Timmias, citizen of Locrus, who are at once a philosopher and a statesman, and to you, Cretius, whom all Athenians know to be similarly accomplished, and to Hermocrates, who is also fitted by nature and education to share in our discourse.

### 2961-961-0005

参考：SOME POEMS OF SOLON WERE RECITED BY THE BOYS

官方：Some poems of Sollen were recited by the boys.

重导出 all：Some poems of Soren were recited by the boys.

### 3570-5694-0018

参考：THE WEARERS OF UNIFORMS AND LIVERIES MAY BE ROUGHLY DIVIDED INTO TWO CLASSES THE FREE AND THE SERVILE OR THE NOBLE AND THE IGNOBLE

官方：The warros of uniforms and livery may be roughly divided into two classes: the free and the servile, or the noble and the ignoble.

重导出 all：The warros of uniforms and liveries may be roughly divided into two classes: the free and the servile, or the noble and the ignoble.

### 4446-2273-0000

参考：HILDA WAS VERY NICE TO HIM AND HE SAT ON THE EDGE OF HIS CHAIR FLUSHED WITH HIS CONVERSATIONAL EFFORTS AND MOVING HIS CHIN ABOUT NERVOUSLY OVER HIS HIGH COLLAR

官方：Hilda was very nice to him, and he sat on the edge of his chair, flushed with his conversational efforts, and moving his chin about nervously over his high collar.

重导出 all：Hilda was very nice to him, and he sat on the edge of his chair, flushed with his conversational efforts and moving his chin about nervously over his high collar.

### 4992-41797-0013

参考：SHE MAKES EFFORT AFTER EFFORT TREMBLING WITH EAGERNESS AND WHEN SHE FAILS TO REPRODUCE WHAT SHE SEES SHE WORKS HERSELF INTO A FRENZY OF GRIEF AND DISAPPOINTMENT

官方：She makes effort after effort, trembling with eagerness, and when she fails to reproduce what she sees, she works herself into a frenzy of grief and disappointment.

重导出 all：She makes effort after effort, trembling with eagerness. And when she fails to reproduce what she sees, she works herself into a frenzy of grief and disappointment.

### 5142-33396-0032

参考：THE FARMER SAT GLOOMILY ON THE BENCH AND WOULD NOT EAT AND YOU CANNOT WONDER FOR HE SAW US PUTTING POTFULS OF HIS GOOD BEEF AND BASKET LOADS OF BREAD INTO OUR BIG MOUTHS

官方：The farmer sat gloomily on the bench and would not eat. And you cannot wonder, for he saw us putting potfuls of his good beef and bascule loads of bread into our big mouths.

重导出 all：The farmer sat gloomily on the bench and would not eat. And you cannot wonder, for he saw us putting potfuls of his good beef and bascule loads of bread into our big mouths.

### 5142-36377-0023

参考：YOU WERE QUITE RIGHT TO SAY NO AMBROSE BEGAN NEVER SMOKE WITH JOHN JAGO HIS CIGARS WILL POISON YOU

官方：You were quite right to say no. Ambrose began, "Never smoke with John Iago. His cigars will poison you."

重导出 all：You were quite right to say no. Ambrose began, "Never smoke with Johnnie. Ago his cigars will poison you."

### 61-70968-0037

参考：WHAT IS YOUR NAME LORDING ASKED THE LITTLE STROLLER PRESENTLY

官方：What is your name, Lording? Asked the little stroller presently.

重导出 all：What is your name, Lording?" asked the little stroller presently.

### 61-70968-0047

参考：MASTER MONCEUX THE SHERIFF OF NOTTINGHAM WAS MIGHTILY PUT ABOUT WHEN TOLD OF THE RIOTING

官方：Master Monse, the sheriff of Nottingham, was mightily put about when told of the rioting.

重导出 all：Master Monceau, the sheriff of Nottingham, was mightily put about when told of the rioting.

### 7176-92135-0000

参考：HE IS A WELCOME FIGURE AT THE GARDEN PARTIES OF THE ELECT WHO ARE ALWAYS READY TO ENCOURAGE HIM BY ACCEPTING FREE SEATS FOR HIS PLAY ACTOR MANAGERS NOD TO HIM EDITORS ALLOW HIM TO CONTRIBUTE WITHOUT CHARGE TO A SYMPOSIUM ON THE PRICE OF GOLF BALLS

官方：He is a welcome figure at the garden parties of the elect, who are always ready to encourage him by accepting free seats for his play. Actor managers nod to him, editors allow him to contribute without charge to a symposium on the price of golf balls.

重导出 all：He is a welcome figure at the garden parties of the elect, who are always ready to encourage him by accepting free seats for his play. Actor managers nod to him. Editors allow him to contribute without charge to a symposium on the price of golf balls.

### 7176-92135-0009

参考：AND I SHOULD BEGIN WITH A SHORT HOMILY ON SOLILOQUY

官方：And I should begin with a short homily on, the little quay.

重导出 all：And I should begin with a short homily on, celibacy.

### 7729-102255-0028

参考：PRIVATE PERSONS WHO HAD LEASED THE FREE STATE HOTEL VAINLY BESOUGHT THE VARIOUS AUTHORITIES TO PREVENT THE DESTRUCTION OF THEIR PROPERTY

官方：Private persons who at least the free state hotel vainly besought the various authorities to prevent the destruction of their property.

重导出 all：Private persons who at least the Free State Hotel vainly besought the various authorities to present the destruction of their property.

### 8455-210777-0057

参考：BUT IN THEIR SELECTION OF A CONSTITUTION THE BRITANNULISTS HAVE UNFORTUNATELY ALLOWED THEMSELVES BUT ONE DELIBERATIVE ASSEMBLY AND HENCE HAVE SPRUNG THEIR PRESENT DIFFICULTIES

官方：But in their selection of a constitution, the Briton newists have unfortunately allowed themselves but one deliberate assembly, and hence have sprung their present difficulties.

重导出 all：But in their selection of a constitution, the Briton newists have unfortunately allowed themselves but one deliberate assembly, and hence has sprung their present difficulties.

### 8463-294828-0010

参考：NEVER DID HE OBJECT TO BUCKLING UP HIS SUITCASE FOR ANY COUNTRY WHATEVER CHINA OR THE CONGO NO MATTER HOW FAR OFF IT WAS

官方：Never did he object to buckling up his suitcase for any country, whatever—China or the Congo—no matter how far off it was.

重导出 all：Never did he object to buckling up his suitcase for any country whatever—China or the Congo—no matter how far off it was.

### 8555-284449-0007

参考：NOW THEN LET'S ENTER THE CITY AN ENJOY THE GRAND FEAST THAT'S BEING COOKED I'M NEARLY STARVED MYSELF FOR THIS CONQUERIN KINGDOMS IS HARD WORK

官方：Now then, let's enter the city and enjoy the great feast that's being cooked. I'm nearly starving myself. For this conquering kingdom's is hard work.

重导出 all：Now then, let's enter the city and enjoy the great feast that's being cooked. I'm nearly starving myself for this conquering kingdom's as hard work.

### 908-31957-0006

参考：OPEN THY HEART WIDE AND FOLD WITHIN THE WET WINGS OF THY DOVE

官方：Open thy heart wide, and fold within the wet wings of thy dove.

重导出 all：Open thy heart wide and fold within the wet wings of thy dove.

### BAC009S0913W0228

参考：更  好  地  适应  了  新  形势  的  需要

官方：\udc8d

重导出 all：更好地适应了新形势的需要。

### BAC009S0914W0128

参考：而  就  昨日  六部  委  松绑  楼市  限  外  令  的  情况  来看

官方：而就昨日六部委松绑楼市、限外令的情况来看。

重导出 all：而就昨日六部委松绑楼市限外令的情况来看。

### BAC009S0914W0406

参考：来源  时光  网  美国  时间  本  周一

官方：来源：时光网，美国时间本周一。

重导出 all：来源时光网，美国时间本周一。

### BAC009S0914W0440

参考：并  被  店主  及  店员  辱骂  围殴

官方：并被店主及店员辱骂为“O”。

重导出 all：并被店主及店员辱骂为殴。

### BAC009S0915W0354

参考：让  朱  婷  最  佳  球员  的  身份  和  价值  再度  彰显

官方：让朱婷最佳球员的身份和价值再度彰显。

重导出 all：让朱婷最佳球员的身份和价值，再度彰显。

### BAC009S0912W0242

参考：重点  治理  农村  土壤  饮用  水水  源地  污染

官方：重点治理农村土壤、饮用水水源地污染。

重导出 all：重点治理农村土壤饮用水水源地污染。

### BAC009S0912W0169

参考：约  六成  网友  看涨  全国  的  商品  房价  格

官方：约六成网友看涨全国的商品房价格。

重导出 all：约六成网友看涨全国的商品房价格。

### BAC009S0906W0323

参考：车  企  诚  待  转化  互联  网  思维  以及  加强  互联  网  安全  管控  等

官方：车企，承担转化互联网思维，以及加强互联网，安全管控等。

重导出 all：车企，承担转化互联网思维以及加强互联网，安全管控等。

### BAC009S0906W0210

参考：新增  千亿  斤  粮食  生产  能力  建设  工程

官方：新增千亿斤粮食生产能力建设工程。

重导出 all：新增千亿斤粮食生产能力，建设工程。

### BAC009S0908W0311

参考：美  的  吸取  了当  年  大跃进  的  教训

官方：美的吸取了当年大跃进的教训。

重导出 all：美帝吸取了当年大跃进的教训。

### BAC009S0908W0389

参考：但  喜欢  欧美  音乐  爱吃  西红柿  意大利  面

官方：但喜欢欧美音乐，爱吃西红柿意大利面。

重导出 all：\udc8d

### BAC009S0907W0421

参考：虽然  未知  去向

官方：сть

重导出 all：虽然未知去向。

### BAC009S0905W0333

参考：小天才  电话  手表  等  国内  几  个  大  品牌  都  有

官方：小天才电话手表，等国内几个大品牌都有。

重导出 all：小天才电话手表等国内几个大品牌都有。

### BAC009S0905W0327

参考：专家  提醒  相关  的  产品  认证  也是  消费  者  必须  要  关注  的

官方：专家提醒：相关的产品认证也是消费者必须要关注的。

重导出 all：专家提醒：，相关的产品认证也是消费者必须要关注的。

### BAC009S0769W0399

参考：志愿  者  除了  要  具备  流利  的  英语  交流  能力  外

官方：志愿者除了要具备流利的英语交流能力外，

重导出 all：志愿者除了要具备流利的英语交流能力外。

### BAC009S0903W0433

参考：温岭  鞋厂  倒塌  事故  已  一四  人  遇难  鞋厂  老板  被控  制

官方：温岭鞋厂倒塌事故，已四人遇难，鞋厂老板被控制。

重导出 all：温岭鞋厂倒塌事故，一四人遇难，鞋厂老板被控制。

### BAC009S0904W0359

参考：结果  等到  的  消息  是  她  不  能  去世  界  杯

官方：结果等到的消息是：“他不能去世界杯。”

重导出 all：结果等到的消息是，他不能去世界杯。

### BAC009S0904W0252

参考：淘汰  报废  高  耗能  老旧  农业  机械

官方：淘汰报废，高耗能老旧农业机械。

重导出 all：淘汰报废、高耗能老旧农业机械。
