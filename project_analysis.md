Transformer Tabanlı Şerit Tespiti Yöntemleri
Otonom sürüşte şerit tespiti (lane detection), aracın şerit çizgilerini algılayarak konumunu güvenli biçimde korumasını sağlayan kritik bir görevdir. Geleneksel derin öğrenme tabanlı şerit tespiti genellikle konvolüsyonel sinir ağları ve segmentasyon yaklaşımına dayanırken, son dönemde Transformer mimarileri ve dil-modeli yaklaşımları bu alana yenilikçi çözümler getirmiştir. Aşağıda, transformer tabanlı veya dil modeli temelli şerit tespiti yöntemlerine odaklanan ve açık kaynak kod sunan önemli akademik çalışmalar özetlenmiştir. (Not: LaneLM gibi yalnızca teori sunup kod paylaşmayan çalışmalar bu listeye dahil edilmemiştir.)
İki Boyutlu (2D) Şerit Tespitinde Transformer Yaklaşımları
LaneATT (CVPR 2021)
Makale Başlığı: Keep Your Eyes on the Lane: Real-time Attention-guided Lane Detection (Lucas Tabelini ve ark.)
openaccess.thecvf.com
Yayın Tarihi: 2021 (CVPR 2021 konferansı).
Mimari ve Yöntem: Anchor tabanlı tek aşamalı bir derin şerit tespit modeli önerilmiştir. Model, nesne tespitine benzer şekilde önceden tanımlı anchor bölgelerinde özellik havuzu (feature pooling) yapar ve anchor tabanlı özgün bir dikkat mekanizması (attention) ile küresel sahne bilgisini toplar
openaccess.thecvf.com
openaccess.thecvf.com
. Hafif bir CNN omurgası kullanmasına rağmen, anchor merkezli bu dikkat yaklaşımı sayesinde perdelenmiş veya eksik şerit çizgilerini küresel bağlam yardımıyla daha iyi tahmin edebilmektedir
openaccess.thecvf.com
.
Açık Kaynak Kod: Resmî açık kaynak kod ve ön eğitimli modeller GitHub deposu üzerinden sunulmuştur (lucastabelini/LaneATT)
openaccess.thecvf.com
.
Benchmark Sonuçları: LaneATT modeli TuSimple, CULane ve LLAMAS gibi üç yaygın veri setinde kapsamlı olarak değerlendirilmiş ve mevcut en iyi yöntemleri hem doğruluk hem de hız açısından geride bırakmıştır
openaccess.thecvf.com
. Örneğin CULane veri setinde hem daha yüksek F1 skoru elde ederken, modelin gerçek zamanlı çalışacak kadar hızlı olduğu gösterilmiştir (~250 FPS hız ile, önceki en iyi modele kıyasla hesaplama yükünü neredeyse 10 kat azaltmıştır)
openaccess.thecvf.com
.
Katkı ve Farklar: LaneATT’nin en büyük yeniliği, şerit tespitine dikkat mekanizmasını entegre eden ilk anchor tabanlı çerçeve olmasıdır. Bu sayede model, görüntüdeki şeritlerin konumlarını diğer şeritlerle ilişkili şekilde küresel ölçekte değerlendirerek çıkarım yapar ve özellikle örtülme (occlusion) veya silik şerit çizgileri durumlarında güçlü bir performans sergiler
openaccess.thecvf.com
. Ayrıca son derece hafif ve hızlı olmasıyla öne çıkar: Önceki çalışmalarda gereken karmaşık ardıl işlemlere (örn. yoğun post-processing, NMS) ihtiyaç duymadan, gerçek zamanlı çalışırken o dönemin en yüksek doğruluklarından birine ulaşmıştır
openaccess.thecvf.com
. Bu yönleriyle LaneATT, pratik otonom sürüş uygulamaları için önemli bir adım olmuştur.
LSTR – Lane Shape Prediction with Transformers (WACV 2021)
Makale Başlığı: End-to-End Lane Shape Prediction with Transformers (Ruijin Liu ve ark.)
ar5iv.labs.arxiv.org
Yayın Tarihi: 2021 (WACV 2021 konferansı).
Mimari ve Yöntem: LSTR, şerit tespit problemini bir dil modeli gibi dizisel bir çıktı yerine doğrudan parametrik bir eğri olarak çözen ilk yöntemlerdendir. Tamamen transformer tabanlı bir uçtan uca ağ kullanarak, görüntüden her bir şeridin polinom eğri parametrelerini doğrudan tahmin eder
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Bu mimari, küresel bağlam ve şeritlerin uzun-ince yapısını yakalamak için kendine dikkat mekanizmalarından (self-attention) yararlanır. Ağın çıktıları, her bir şeride ait parametre gruplarıdır ve klasik yöntemlerin aksine yoğun bir piksel segmentasyonu yerine bu parametrelerin öğrenilmesini hedefler
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
.
Açık Kaynak Kod: Resmî kod deposu mevcuttur (liuruijin17/LSTR) ve deneylerin tekrarlanabilmesi için paylaşılmıştır
ar5iv.labs.arxiv.org
.
Benchmark Sonuçları: LSTR modeli, TuSimple şerit tespiti benchmark’ında %96.18 doğruluk gibi son derece yüksek bir skora ulaşarak o dönemki duruma göre yeni bir seviye belirlemiştir
github.com
. Model sadece başarılı değil, aynı zamanda son derece hafiftir: toplam parametre sayısı 0.76M mertebesinde ve hesaplama yükü 574 milyon MAC civarındadır, bu da gerçek zamanlı uygulamalarda büyük avantaj sağlar
github.com
github.com
. En yakın rakiplerine benzer veya daha yüksek doğruluğa ulaşırken model boyutu ve hız açısından en iyilerden biri olduğunu kanıtlamıştır.
Katkı ve Farklar: Bu çalışmanın yenilikçi katkısı, şerit tespitini tek adımlı bir regresyon problemi olarak ele almasıdır. LSTR, çıktı olarak piksellerden oluşan bir maskeyi değil, her bir şeridi tanımlayan matematiksel parametreleri verir. Bunu mümkün kılmak için literatürde ilk defa transfomer bloğunu kullanarak şerit noktaları arasındaki uzun menzilli ilişkileri ve global sahne bilgisini öğrenmiştir
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Model, çıktı şerit parametrelerini gerçek şeritlerle eşleştirmek için Hungarian eşleştirme tabanlı bir kayıp fonksiyonu kullanır ve bu sayede bir görüntüdeki her şeridi tekil bir hedef olarak öğrenir
ar5iv.labs.arxiv.org
. Bu tasarımın önemli bir avantajı, son işlem olarak Non-Maximum Suppression (NMS) gereksinimini ortadan kaldırması ve daha basit bir çıkarım süreci sunmasıdır
ar5iv.labs.arxiv.org
. Sonuç olarak LSTR, hem yüksek doğruluk hem de en hızlı inference özelliklerini bir araya getirerek, şerit tespitinde transformer kullanımının etkinliğini göstermiştir.
LaneFormer (AAAI 2022)
Makale Başlığı: Laneformer: Object-Aware Row-Column Transformers for Lane Detection (Jianhua Han ve ark.)
arxiv.org
Yayın Tarihi: 2022 (AAAI 2022 konferansı).
Mimari ve Yöntem: LaneFormer, şerit algılama için özelleştirilmiş bir encoder-decoder Transformer mimarisi sunar. Başlıca yenilik, satır ve sütun boyutlu kendiliğinden dikkat mekanizmalarının getirilmesidir: Encoder aşamasında, her bir piksel özelliği üzerinde “row-attention” (aynı yatay satırdaki pikseller arası etkileşim) ve “column-attention” (aynı dikey sütundaki pikseller arası etkileşim) işlemleri uygulanır
cdn.aaai.org
. Bu sayede model, şerit çizgilerinin görüntüdeki geometrik şekillerini daha etkin yakalar: Ardışık satırlarda aynı şeride ait piksellerin yakın konumda olacağı bilgisi satır-dikkat ile işlenirken; farklı şeritlerin ayrı sütunlarda bulunması durumu sütun-dikkat ile ayrıştırılır (Şekil 1’de şematize edilmiştir)
cdn.aaai.org
. Ayrıca LaneFormer, ortamdaki nesne algılamalarını da Transformer’e entegre eden bir yaklaşıma sahiptir. Öncül bir nesne tespit modeliyle bulunan araç/yaya gibi nesnelerin bulanık kutu konumları (bbox), transformer’ın dikkat mekanizmasında Key olarak, bu kutulardan çıkarılan ROI özellik vektörleri ise Value olarak beslenir
arxiv.org
. Böylece şerit tespiti sırasında model, etraftaki nesnelerin varlığını ve konumunu da göz önüne alarak (ör. araçların hemen yanında şerit olma olasılığı gibi) daha bağlamsal bir çıkarım yapar
arxiv.org
.
Açık Kaynak Kod: Resmî kod paylaşılmıştır (Huawei Noah’s Ark Lab – Codes for Lane Detection reposu altında) ve deneylerin tekrarı için erişilebilir durumdadır
researchgate.net
.
Benchmark Sonuçları: LaneFormer, CULane veri setinde %77.1 F1 skoru elde ederek o zamana kadarki en yüksek performansı sergilemiştir
cdn.aaai.org
. Aynı model, TuSimple veri setinde de %96.8 doğruluk başarımına ulaşmış, böylece 2D şerit tespitinde hem şehir içi karmaşık sahnelerde (CULane) hem de otoyol senaryolarında (TuSimple) üstünlüğünü kanıtlamıştır
cdn.aaai.org
. Modelin verimliliği de yüksektir; ResNet-50 tabanlı LaneFormer, tek bir GPU üzerinde ~50 FPS hızına varan gerçek zamanlı performans raporlamıştır
cdn.aaai.org
.
Katkı ve Farklar: LaneFormer, şerit algılama problemine uzamsal dikkat (spatial attention) konusunda yeni bir bakış açısı getirmiştir. Özellikle, ilk defa satır ve sütun yönelimli dikkat bileşenleri kullanılarak şeritlerin uzunlamasına yapısı ve paralelliği etkin şekilde modele dahil edilmiştir. Bu yapı, global-bağlamsal öğrenme ile yerel geometrik kısıtları birleştirerek, NMS veya kümelenmiş post-processing adımlarına ihtiyaç duymadan doğrudan doğruya şerit tespiti yapabilmeyi sağlar
cdn.aaai.org
ar5iv.labs.arxiv.org
. Dahası, nesne farkındalığının dikkat mekanizmasına entegre edilmesi, otonom sürüş sahnelerinde sıkça rastlanan araç trafiği ve engellemeler altında modelin dayanıklılığını artırmıştır. LaneFormer ile gösterilen bir diğer önemli çıktı, transformer tabanlı bir modelin optimize implementasyon ile gerçek zamanlı hızlara yakın çalışabileceğidir (yaklaşık 48–53 FPS)
cdn.aaai.org
. Bu yönüyle LaneFormer, hem akademik hem endüstriyel açıdan 2D şerit tespitinde güçlü bir temel oluşturmuştur.
CondLane (CondLSTR, ICCV 2023)
Makale Başlığı: Generating Dynamic Kernels via Transformers for Lane Detection (Ziye Chen ve ark.)
openaccess.thecvf.com
openaccess.thecvf.com
Yayın Tarihi: 2023 (ICCV 2023 konferansı).
Mimari ve Yöntem: CondLane (makalede CondLSTR olarak da anılır), şerit tespiti için dinamik evrişim çekirdekleri üreten bir transformer mimarisi sunar. Bu yöntemde bir transformer bloğu, görüntüdeki her bir şerit çizgisi için özel bir konvolüsyon çekirdeğini dinamik olarak oluşturur; ardından bu çekirdek, özellik haritası üzerinde katmanlı evrişim şeklinde uygulananarak ilgili şeridi tespit eder
openaccess.thecvf.com
. Klasik yaklaşımda dinamik çekirdekler genellikle sadece şerit başlangıç noktası gibi sınırlı bir bölgeden türetilirken, CondLane’de transformer global şerit bilgisini öğrenerek çekirdekleri üretir. Bu sayede üretilen filtreler, şeridin tüm eğrisel yapısını (uzun ve kıvrımlı olsa dahi) hesaba katar ve şeritlerin çatallanması, kesişmesi veya araçlarla örtülmesi durumlarında bile sağlam bir tespit gerçekleştirir
openaccess.thecvf.com
openaccess.thecvf.com
. Başka bir deyişle, model belirli bir geometrik form varsayımına (örneğin yalnızca düz çizgi ya da polinom) dayanmaz; bunun yerine veriden öğrenilen esnek çekirdeklerle her türlü şerit topolojisine uyum sağlar
openaccess.thecvf.com
openaccess.thecvf.com
.
Açık Kaynak Kod: Bu çalışmanın PyTorch tabanlı resmi uygulaması GitHub üzerinde paylaşılmıştır (czyczyyzc/CondLSTR deposu) ve araştırmacıların kullanımına sunulmuştur. README dokümanında yöntemin çerçevesi ve kullanım adımları ayrıntılı olarak verilmiştir
github.com
.
Benchmark Sonuçları: CondLane yöntemi, mevcut en iyi yöntemleri önemli farklarla geride bırakarak yeni rekorlar kırmıştır. Örneğin OpenLane (3D şeritler içeren açık ortam veri seti) üzerinde F1 skoru 63.40 elde ederek önceki en iyi sonucu +4.30 puan geliştirmiştir. Benzer şekilde CurveLanes (zorlayıcı eğri şerit veri seti) üzerinde F1 skoru 88.47 ile bir önceki en iyi yöntemi +2.37 puan aşmıştır
openaccess.thecvf.com
. Bu belirgin iyileştirmeler, özellikle karmaşık yapılı ve dönüşlü şeritlerin olduğu senaryolarda CondLane’in üstün performans gösterdiğine işaret etmektedir.
Katkı ve Farklar: CondLane, şerit tespitinde özel bilgiye dayalı modellerden (ör. belirli polinom/spline varsayımları) genel öğrenilebilir modellere geçişi simgeleyen bir çalışmadır. Transformer tabanlı dinamik çekirdek üretimi sayesinde, model şerit şekillerinin global yapısını doğrudan öğrenip evrişim filtrelerine yansıtır. Bu yaklaşım, özellikle çatal yapan veya birden fazla kola ayrılan şeritler, yoğun biçimde paralel giden şeritler ve araçlarca kısmen gizlenmiş şeritler gibi durumlarda, önceki sabit çekirdek kullanan yöntemlere kıyasla belirgin bir avantaj sağlar
openaccess.thecvf.com
openaccess.thecvf.com
. CondLane’in bir diğer farkı, şerit tespit sürecini transformer ile öğrenilebilir bir alt ağa dönüştürmesidir: Bu sayede model, sahne içindeki her şeridi bir sıra dizisi ya da maskeden ziyade, o şeride özgü bir filtre olarak temsil eder. Bu yenilikçi bakış açısı, dil modellerindeki sıra-tabanlı yaklaşım ile görüntü tabanlı evrişimsel yaklaşımlar arasında bir köprü kurarak literatüre katkı sunmuştur. Sonuç olarak CondLane, 2B şerit tespitinde hem esneklik hem de doğruluk açısından önemli bir ilerlemeyi temsil etmektedir.
Üç Boyutlu (3D) Şerit Tespitinde Transformer Yaklaşımları
PersFormer (ECCV 2022)
Makale Başlığı: PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark (Li Chen ve ark.)
ar5iv.labs.arxiv.org
Yayın Tarihi: 2022 (ECCV 2022, sözlü sunum).
Mimari ve Yöntem: PersFormer, monoküler kamera görüntüsünden 3B şeritleri tespit eden uçtan uca bir modeldir. Ana yeniliği, perspektiften kuşbakışına (BEV) özellik dönüşümünü bir Transformer modülü ile gerçekleştirmesidir
ar5iv.labs.arxiv.org
. Modelin “perspective transformer” adı verilen bileşeni, kamera iç ve dış parametrelerini referans alarak ön-iz görüntüdeki lokal bölgeleri kuşbakışı düzleme aktarır; bunu yaparken çoklu başlıkli dikkat mekanizması kullanarak, her BEV konumunun ilgili olduğu ön-iz bölgesini öğrenir
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Bu sayede geleneksel sabit dönüştürme (ör. Inverse Perspective Mapping) yöntemlerinde yaşanan derinlik hizalama hataları en aza indirilir. PersFormer, aynı zamanda birleşik bir 2B/3B anchor tasarımı kullanır ve 2B ile 3B şerit tespitini eş-zamanlı yapan çok-görevli bir ağdır; ortak bir öğrenme ile 2B ve 3B çıktılar birbirini destekleyerek daha tutarlı özellikler elde edilmesini sağlar
ar5iv.labs.arxiv.org
github.com
. Özetle, kamera görüntüsünden doğrudan 3B uzayda şeritleri çıkarabilmek için hem geometrik dönüşümü öğrenen, hem de 3B verisizlik problemini (yüksekliğin belirsizliği) kameranın zemin düzlemine göre çözen bir mimari sunulmuştur.
Açık Kaynak Kod: Projenin kodları ve eğitimli model ağırlıkları açık kaynak olarak paylaşılmıştır (OpenDriveLab/PersFormer_3DLane GitHub deposu)
ar5iv.labs.arxiv.org
. Ayrıca çalışma kapsamında otonom sürüş için kapsamlı bir 3B şerit veri seti olan OpenLane de yayınlanmıştır (200k görüntü karesi, ~880k şerit örneği içerir)
ar5iv.labs.arxiv.org
.
Benchmark Sonuçları: PersFormer, sunulan OpenLane veri setinde ve Apollo 3D Sürüm (simülasyon) veri setinde mevcut yöntemleri kayda değer farkla geride bırakmıştır
ar5iv.labs.arxiv.org
. Örneğin, OpenLane üzerinde F1 skoru ~53 civarında elde edilerek, bir önceki en iyi yöntem olan 3D-LaneNet’in (~44 F1) oldukça üzerine çıkılmıştır
github.com
. Apollo simülasyon ortamında ve ONCE-3DLanes gibi diğer benchmark’larda da benzer üstünlükler raporlanmıştır. Ayrıca PersFormer, OpenLane veri setinin 2B şerit tespiti kısmında da çağdaş 2B yöntemlerle kıyaslanabilir bir doğruluk yakalayarak çok yönlülüğünü göstermiştir
ar5iv.labs.arxiv.org
.
Katkı ve Farklar: PersFormer’in en önemli katkısı, görüntü uzayından BEV uzayına öğrenilebilir bir dönüşüm gerçekleştirerek 3B şerit tespitindeki temel sorunlardan birini çözmesidir. Klasik yöntemlerdeki plan projeksiyon varsayımının ötesine geçerek, kamera görüş açısı değişimleri, yol eğimleri gibi durumlarda dahi güvenilir şerit çıkarımı mümkün kılınmıştır
ar5iv.labs.arxiv.org
. Ayrıca bu çalışma ile birlikte yayınlanan OpenLane veri seti, gerçek dünyadan yüksek hacimli 3B şerit verisi sunarak alanda bir standart oluşturmuştur
ar5iv.labs.arxiv.org
. PersFormer modeli, 3B şerit algılama için o güne kadarki en iyi sonuçları sağlamakla kalmayıp, aynı ağ içinde 2B ve 3B algılama yaparak çok-görevli öğrenmenin faydalarını ortaya koymuştur
ar5iv.labs.arxiv.org
ar5iv.labs.arxiv.org
. Bu yönüyle PersFormer, otonom sürüş algı sistemlerinde kameradan 3B anlayış çıkarma problemini pratikçe çözen öncü bir çalışma olarak değerlendirilebilir.
LATR (ICCV 2023)
Makale Başlığı: LATR: 3D Lane Detection from Monocular Images with Transformer (Yueru Luo ve ark.)
arxiv.org
Yayın Tarihi: 2023 (ICCV 2023, sözlü sunum).
Mimari ve Yöntem: LATR, monoküler bir görüntüden 3B şeritleri tespit etmek için tasarlanmış bir transformer tabanlı algılayıcıdır. Bu model, 3B uzaysal bilgisini kullanmak için açıkça bir BEV görüntü oluşturmaya gerek duymadan, görüntü özelliklerini 3B farkındalıklı hale getirmektedir
arxiv.org
. LATR mimarisinde, görüntüden çıkarılan özellikler içerisine, iteratif olarak güncellenen bir sanal 3B zemin düzlemine göre hesaplanan konumsal gömme vektörleri eklenir
arxiv.org
. Ardından model, belirli sayıda öğrenilebilir sorgu (query) tanımlayarak, sorgu-ve-anahtar-değer tabanlı bir çapraz-dikkat mekanizması kurar
arxiv.org
. Her bir sorgu vektörü, 2B görüntüdeki olası bir şerit adayının özelliklerinden üretilir ve bu sorgular, görüntü özellik haritasındaki anahtar-değer ikilileriyle etkileşime girerek doğrudan ilgili şeridin 3B koordinatlarını tahmin eder
arxiv.org
. Bu yaklaşım, özellikle monoküler görüntülerde yaşanan derinlik belirsizliği sorununu azaltmayı hedefler; zira LATR, perspektif görüntüdeki özellikleri doğrudan 3B dünyadaki zemine göre konumlandırarak, kuşbakışı dönüşüm olmaksızın şeritleri konumlandırabilir.
Açık Kaynak Kod: Bu çalışmanın resmi kod deposu (JMoonr/LATR) mevcuttur ve ICCV 2023 itibariyle araştırmacılarla paylaşılmıştır
github.com
. Kod ile birlikte model ağırlıkları ve kullanım örnekleri de sunulmaktadır.
Benchmark Sonuçları: LATR, 3B şerit tespiti alanında yayınlandığı dönemde yeni standartlar belirlemiştir. Apollo (sentetik) veri seti, OpenLane (gerçek dünya) ve ONCE-3DLanes gibi çeşitli benchmark’larda önceki en iyi sonuçları ciddi farklarla aşmıştır
arxiv.org
. Örneğin OpenLane veri setinde LATR, F1 skorunu önceki en iyinin tam +11.4 puan üzerine çıkararak büyük bir performans artışı sağlamıştır (önceki ~%42 F1’den LATR ile ~%53.4 F1’e)
arxiv.org
. Bu kazanımlar, LATR’nin özellikle karmaşık sahnelerde (monoküler kameranın dezavantajlı olduğu derinlik belirsizliklerinde) bile üstün bir kesinlikte şerit tespiti yapabildiğini göstermektedir.
Katkı ve Farklar: LATR’nin geliştirilmesiyle monoküler kamera görüntülerinden 3B şerit tespitine yönelik yeni bir paradigma ortaya konmuştur. PersFormer gibi yöntemler BEV dönüşümü kullanırken, LATR bunun yerine 3B bilgisini doğrudan dikkat mekanizmasına entegre ederek perspektif kayıpları ve hizalama hatalarını minimize etmiştir
arxiv.org
. Modelde tanımlanan öğrenilebilir sorgular ve dinamik 3B pozisyonel gömme tekniği, her bir şeridin 3B uzaydaki şeklini kademeli olarak iyileştirerek tahmin etme imkânı tanır
arxiv.org
. Bu yenilikler sayesinde LATR, 3B şerit algılama performansında çarpıcı bir sıçrama elde etmiş ve derin öğrenme modellerinin 3B uzayda bağlamsal bilgi kullanımı konusunda etkileyici bir örnek teşkil etmiştir (OpenLane benchmark’ındaki +11.4 puanlık F1 artışı buna somut bir kanıttır)
arxiv.org
.
Kaynaklar: Yukarıda anılan çalışmaların bilgileri ilgili makalelerin özetlerinden ve raporladıkları deneysel sonuçlardan derlenmiştir. Her bir başlık altında verilen referanslar (【x†Ly-Lz】 biçiminde) doğrudan ilgili makaleye veya proje sayfasına ait olup, daha fazla detay için incelenmeleri önerilir. Bu derlemede yalnızca açık kaynak kodu sağlanmış ve deneysel olarak doğrulanmış yöntemler ele alınmıştır.
Alıntılar

Keep Your Eyes on the Lane: Real-Time Attention-Guided Lane Detection

https://openaccess.thecvf.com/content/CVPR2021/papers/Tabelini_Keep_Your_Eyes_on_the_Lane_Real-Time_Attention-Guided_Lane_Detection_CVPR_2021_paper.pdf

Keep Your Eyes on the Lane: Real-Time Attention-Guided Lane Detection

https://openaccess.thecvf.com/content/CVPR2021/papers/Tabelini_Keep_Your_Eyes_on_the_Lane_Real-Time_Attention-Guided_Lane_Detection_CVPR_2021_paper.pdf

Keep Your Eyes on the Lane: Real-Time Attention-Guided Lane Detection

https://openaccess.thecvf.com/content/CVPR2021/papers/Tabelini_Keep_Your_Eyes_on_the_Lane_Real-Time_Attention-Guided_Lane_Detection_CVPR_2021_paper.pdf

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

GitHub - liuruijin17/LSTR: This is an official repository of End-to-end Lane Shape Prediction with Transformers.

https://github.com/liuruijin17/LSTR

GitHub - liuruijin17/LSTR: This is an official repository of End-to-end Lane Shape Prediction with Transformers.

https://github.com/liuruijin17/LSTR

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2203.09830] Laneformer: Object-aware Row-Column Transformers for Lane Detection

https://arxiv.org/abs/2203.09830
Laneformer: Object-Aware Row-Column Transformers for Lane Detection

https://cdn.aaai.org/ojs/19961/19961-13-23974-1-2-20220628.pdf

[2203.09830] Laneformer: Object-aware Row-Column Transformers for Lane Detection

https://arxiv.org/abs/2203.09830

(PDF) Laneformer: Object-aware Row-Column Transformers for ...

https://www.researchgate.net/publication/359367973_Laneformer_Object-aware_Row-Column_Transformers_for_Lane_Detection
Laneformer: Object-Aware Row-Column Transformers for Lane Detection

https://cdn.aaai.org/ojs/19961/19961-13-23974-1-2-20220628.pdf
Laneformer: Object-Aware Row-Column Transformers for Lane Detection

https://cdn.aaai.org/ojs/19961/19961-13-23974-1-2-20220628.pdf
Laneformer: Object-Aware Row-Column Transformers for Lane Detection

https://cdn.aaai.org/ojs/19961/19961-13-23974-1-2-20220628.pdf

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

GitHub - czyczyyzc/CondLSTR: Code for paper "Generating Dynamic Kernels via Transformers for Lane Detection"

https://github.com/czyczyyzc/CondLSTR

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

Generating Dynamic Kernels via Transformers for Lane Detection

https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Generating_Dynamic_Kernels_via_Transformers_for_Lane_Detection_ICCV_2023_paper.pdf

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

GitHub - OpenDriveLab/PersFormer_3DLane: [ECCV 2022 Oral] Perspective Transformer on 3D Lane Detection

https://github.com/OpenDriveLab/PersFormer_3DLane

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

GitHub - OpenDriveLab/PersFormer_3DLane: [ECCV 2022 Oral] Perspective Transformer on 3D Lane Detection

https://github.com/OpenDriveLab/PersFormer_3DLane

[2203.11089] PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark

https://ar5iv.labs.arxiv.org/html/2203.11089

[2308.04583] LATR: 3D Lane Detection from Monocular Images with Transformer

https://arxiv.org/abs/2308.04583

[2308.04583] LATR: 3D Lane Detection from Monocular Images with Transformer

https://arxiv.org/abs/2308.04583

[2308.04583] LATR: 3D Lane Detection from Monocular Images with Transformer

https://arxiv.org/abs/2308.04583

GitHub - JMoonr/LATR: [ICCV2023 Oral] LATR: 3D Lane Detection from Monocular Images with Transformer

https://github.com/JMoonr/LATR

[2308.04583] LATR: 3D Lane Detection from Monocular Images with Transformer

https://arxiv.org/abs/2308.04583

[2308.04583] LATR: 3D Lane Detection from Monocular Images with Transformer

https://arxiv.org/abs/2308.04583

Makaleyi incelediğimde, LaneLM isimli bu çalışma açıkça Transformer mimarisine dayalı bir yaklaşımdır ve özellikle dil modeli benzeri bir yapı ile lane detection (şerit tespiti) görevini çözmeyi hedeflemektedir. Aşağıda bunu adım adım, sade bir dille açıklayayım:

🌐 Bu Makale Ne Yapıyor?

LaneLM, şerit tespiti problemini klasik “görüntüden şekil bulma” yerine bir dil anlama ve üretme problemi gibi ele alıyor. Yani:

Her bir şerit çizgisi, bir kelime dizisi (token sequence) gibi temsil ediliyor.

Model, görüntüdeki şeritleri, tıpkı bir dil modeli gibi, sırayla tahmin ediyor (örneğin: bir kelimeden sonra hangisi gelir? → bir noktadan sonra şerit nereden geçer?).

Bu süreci yönetmek için hem görüntü özelliğini çıkaran bir encoder hem de Transformer decoder kullanan bir dil modeli içeriyor

preprints202504.1582.v1

.

🧠 Kullanılan Mimariler Nelerdir?
1. Transformer Decoder:

Ana şerit tahmini bu bileşende yapılıyor.

Her şerit noktası (keypoint), bir kelime gibi vektöre dönüştürülüyor.

Bu diziler, dil modeli gibi sırayla işleniyor.

Causal attention + cross-attention ile çalışan bir Transformer decoder var

preprints202504.1582.v1

.

2. Cross-Attention Mekanizması:

Görsel bilgi (resimdeki alanlar) ile dilsel bilgi (şerit noktaları dizisi) arasında bağ kuruluyor.

Dikkat (attention) hesaplarında:

Query (sorgular) → keypoint token’lar

Key & Value (anahtar-değer) → görüntüden çıkarılmış yama (patch) vektörleri

Bu yapı tam anlamıyla görsel dil modeli (VLM) mimarisine benziyor.

3. Visual Encoder (CNN + Patch Embedding):

Görüntü önce klasik bir CNN ile işleniyor (ResNet18, DLA34 gibi).

Elde edilen çok katmanlı öznitelikler, ViT benzeri yama yama (patch) vektörlere dönüştürülüyor.

Görsel patch’lere pozisyonel kodlamalar da ekleniyor

preprints202504.1582.v1

.

🧭 Transformer Nerede Kullanılıyor?

Şeritleri bir dizi olarak çıkarma süreci, tamamen Transformer decoder üzerinden yürüyor.

Görsel özelliklerle etkileşim cross-attention ile sağlanıyor (her şerit noktası, görselden hangi bölgeyle ilgiliyse oraya odaklanıyor).

Modelin şerit üretimi autoregressive yani “bir sonraki noktayı sırayla tahmin etme” temelli.

Yani buradaki Transformer kullanımı, klasik ViT gibi değil, dil modeli (language modeling) tarzı bir Transformer.

🎯 LaneATT gibi çalışmalardan farkı ne?
Özellik	LaneATT	LaneLM
Dikkat Mekanizması	Anchor tabanlı attention	Language-like cross-attention
Şerit Temsili	Koordinat regresyonu (yerel)	Token dizisi olarak (kelime gibi)
Transformer Kullanımı	Görsel attention (resim içinde)	Görsel-dil arasında Transformer decoder
Girdi Formatı	Görüntü	Görüntü + birkaç nokta (prompt)
Model Amacı	Hızlı, doğrudan şerit çıkarımı	Etkileşimli, prompt ile yönlendirilen
🔎 Özetle Ne Düşünüyorum?

Bu makale, klasik görsel dikkat yerine dil modeli mimarisiyle lane detection yapan ilk çalışmalardan biri. Şunları açıkça yapıyor:

Görüntü → yama dizisi (CNN + patch embedding)

Şeritler → token dizisi (x,y koordinatları → dil token’ı)

Transformer decoder ile şeritleri sırayla tahmin etme

Dikkat mekanizması ile görüntü ve dil birleştirme

Kullanıcıdan gelen prompt'larla yönlendirilebilme (interaktif yapı)

Ancak şunu belirtmek önemli: Henüz açık kaynak kodu yayınlanmamış, dolayısıyla “kanıtlanmış” bir çalışma değil. Kod erişimi olmadan tekrarlanabilirlik sağlanamıyor — bu da bilimsel katkısını sınırlıyor.

İstersen bu modeli örnek alan basit bir “dil modeliyle lane tahmini” deneme kodu fikri çıkarabilirim ya da bu mimariyi gerçek zamanlı bir sistemde nasıl kullanabileceğini konuşabiliriz.

Anladım! Sana LaneLM tarzı bir mimarinin yapısını sadeleştirilmiş, açıklamalı bir PyTorch kod yapısı üzerinden katman katman anlatacağım. Böylece her bileşenin işlevini doğrudan kod üzerinden görüp anlayabileceksin. Aynı zamanda bu mimarinin mantıklı mı, yoksa karmaşık mı olduğunu da teknik açıdan değerlendireceğim.

Hazırlığım tamamlandığında haber vereceğim, bu sırada başka bir şeyle ilgilenmek istersen buradayım!


LaneLM Benzeri Şerit Tespit Modeli – Transformer Tabanlı Mimari
Aşağıda, LaneLM tarzı bir şerit tespit modelinin temel bileşenlerini adım adım inceliyoruz. Her bir katmanı basit bir PyTorch kod parçacığı ile gösterip, neden gerekli olduğunu vurgulayacağız. Bu mimari, bir CNN tabanlı görsel encoder (özellik çıkarıcı) ile görüntüden özellik haritası elde eder, bunu Transformer tabanlı bir decoder (kod çözücü) ile birleştirerek oto-regresif (ardışıl) bir şekilde şerit noktalarını çıkarır
preprints.org
preprints.org
. Temel bileşenler: Görsel encoder, patch embedding (yama gömme), positional encoding (konum kodlaması), Transformer decoder, cross-attention (çapraz dikkat) ve oto-regresif şerit çıkarımı olarak sıralanabilir.
Görsel Encoder (CNN Backbone)
Görsel encoder, girdi görüntüyü alıp daha kompakt bir özellik gösterimine dönüştüren CNN tabanlı bir katmandır. Genellikle ResNet gibi önceden eğitilmiş bir CNN omurgası kullanılır
preprints.org
. Aşağıdaki örnek kodda basit bir CNN ile özellik haritası çıkarıyoruz:
import torch
import torch.nn as nn

class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Basit bir CNN omurgası (backbone) – birkaç evrişim katmanı
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        feature_map = self.features(x)  # Çıktı: B x 128 x (H/8) x (W/8) boyutlarında özellik haritası
        return feature_map
Bu CNN omurgası, görüntüdeki yüksek seviyeli özellikleri yakalar ve daha küçük boyutlu bir özellik haritası üretir. Neden gerekli: Ham piksel girdisini doğrudan bir transformera vermek verimsizdir; bu nedenle CNN, görsel bilgiyi özetleyerek transformerin işlemeyi daha kolay öğrenebileceği bir forma sokar
preprints.org
.
Patch Embedding (Yama Gömme)
CNN'den gelen 2B özellik haritasını, Transformer'ın anlayacağı 1B dizi (sequence) haline getirmek için patch embedding yapılır. Özellik haritası, sabit boyutlu parçalara (patch) bölünüp her parça düzleştirilir ve lineer projeksiyonla bir vektöre dönüştürülür
preprints.org
. Bu, ViT (Vision Transformer) mantığına benzer bir yaklaşımdır.
class PatchEmbed(nn.Module):
    def __init__(self, in_channels=128, patch_size=4, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        # Özellik haritasını parçalara ayırmak için Unfold kullanıyoruz
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # Her yama parçasını embed_dim boyutunda vektöre projekte eden lineer katman
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
    def forward(self, feature_map):
        # feature_map: [B, C, H, W]
        patches = self.unfold(feature_map)           # Çıktı: [B, C*patch_size^2, N_patches]
        patches = patches.transpose(1, 2)            # Şekil değişimi: [B, N_patches, C*patch_size^2]
        tokens = self.proj(patches)                 # Her patch'i embed_dim boyutuna eşle
        return tokens  # Boyut: [B, N_patches, embed_dim]
Yukarıdaki kod, özellik haritasını patch_size x patch_size boyutlu yamalara böler ve her yamayı bir vektör olarak temsil eder. Neden gerekli: Transformer katmanı sabit boyutlu vektör dizilerini giriş olarak alır; patch embedding, CNN’den gelen 2B veriyi bu gerekli 1B dizi formatına dönüştürür
preprints.org
.
Positional Encoding (Konum Kodlaması)
Patch embedding sonucunda elde edilen token vektörleri, uzamsal konum bilgilerini artık içermez çünkü yamaları düzleştirdik. Transformer’ın dizideki her token’ın görüntünün neresinden geldiğini anlaması için konum bilgisi eklemek gerekir
preprints.org
. Bunu ya sinüs-kosinüs fonksiyonlarıyla oluşturulan sabit konum kodlarıyla ya da öğrenilebilir bir parametre vektörüyle yapabiliriz.
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        # Öğrenilebilir konum gömme (max_len uzunluğa kadar)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
    def forward(self, tokens):
        # tokens: [B, N, embed_dim]
        seq_len = tokens.size(1)
        # İlk seq_len kadar konum vektörünü ekle
        tokens = tokens + self.pos_embedding[:, :seq_len, :]
        return tokens
Bu katman, her bir görsel token vektörüne kendi konumuna karşılık gelen ek bir vektör toplar. Neden gerekli: Konum kodlaması olmadan Transformer, dizideki yamaların sıralamasını ya da uzamsal düzenini bilemez; konum bilgisi, her yamanın görüntüdeki yerini modele hissettirir.
Transformer Decoder (Kod Çözücü) Yapısı
Bu mimaride, şerit noktalarını ardışıl bir dizi olarak tahmin etmek için bir Transformer decoder kullanılır
preprints.org
. Decoder, dil modellerine benzer şekilde çalışır: Şu ana kadar üretilen şerit noktalarını girdide alıp sonraki noktayı tahmin eder. Bunu yaparken hem kendi geçmiş çıktılarından (self-attention ile) hem de görüntüden elde edilen token dizisinden (cross-attention ile) bilgi alır. Aşağıda tek bir Transformer decoder katmanının yapısını basitçe gösteren bir kod yer alıyor:
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512):
        super().__init__()
        # 1. Maskeli self-attention: şerit tokenları kendi içinde dikkat mekanizması
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # 2. Cross-attention: görsel bellek (image tokens) üzerinde dikkat
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # 3. İleri beslemeli ağ (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        # Katman normları
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
    def forward(self, tgt_seq, memory):
        # tgt_seq: [T, B, E] (şu ana kadarki şerit tokenlarının embeddings)
        # memory:  [M, B, E] (görsel tokenlar; CNN+patch emb. sonrası)
        # Self-Attention (maskeli, kausal)
        attn_out, _ = self.self_attn(tgt_seq, tgt_seq, tgt_seq, 
                                     attn_mask=None)  # gerçekte kausal maske uygulanır
        tgt_seq = self.norm1(tgt_seq + attn_out)
        # Cross-Attention (görsel bellek üzerinden)
        attn_out2, _ = self.cross_attn(tgt_seq, memory, memory)
        tgt_seq = self.norm2(tgt_seq + attn_out2)
        # Feed-forward network
        ff_out = self.ffn(tgt_seq)
        tgt_seq = self.norm3(tgt_seq + ff_out)
        return tgt_seq
Yukarıdaki DecoderLayer, standart bir Transformer decoder bloğuna benzer şekilde önce self-attention, sonra cross-attention ve ardından bir ileri beslemeli ağ uygular. Self-attention kısmı genellikle gelecek tokenları maskeler (kausal mask) ki model kendi henüz tahmin etmediği ileriki noktaları görmesin. Cross-attention kısmı ise bir sonraki noktayı tahmin ederken görüntüden gelen özelliklere odaklanmayı sağlar
preprints.org
. Neden gerekli: Decoder, şerit noktalarının dizisini üretmek için dil modeli mantığında çalışır; self-attention ile dizinin tutarlılığını sağlar, cross-attention ile görsel bağlamdan yararlanır. Bu sayede model, önceki noktaları ve görüntü bilgisini bir araya getirerek mantıklı bir sonraki nokta üretebilir.
Cross-Attention (Çapraz Dikkat Mekanizması)
Cross-attention, decoder katmanının kritik bir parçasıdır. Decoder’daki sorgu (query) vektörleri şerit tokenlarından gelirken, anahtar (key) ve değer (value) vektörleri görsel encoder tarafından üretilen token dizisinden alınır
preprints.org
. Bu sayede model, her bir şerit noktası tahmininde tüm görüntü özelliklerine bakabilir. Aşağıdaki mini kod parçası cross-attention kullanımını gösterir:
# diyelim ki lane_tokens (T, B, E) ve image_tokens (M, B, E) elimizde var
query = lane_tokens   # sorgu: şerit tokenları (mevcut dizi)
key   = image_tokens  # anahtar: görsel tokenlar
value = image_tokens  # değer: görsel tokenlar
out, attn_weights = cross_attn(query, key, value)
Burada cross_attn bir MultiheadAttention nesnesidir. Sorgu dizisi, o ana kadarki şerit noktalarını temsil eder; anahtar ve değer ise görüntünün tüm patch tokenlarıdır. Sonuç out, sorgu tokenlarının, görüntüdeki hangi bölgelere dikkat ettiğini yansıtarak güncellenmiş temsilidir, attn_weights ise her sorgu tokenının hangi görsel tokenlara ne kadar dikkat verdiğinin ağırlıklarıdır. Neden gerekli: Çapraz dikkat, modelin tahmin edeceği bir sonraki şerit noktası için görüntünün ilgili bölgelerinden bilgi almasını sağlar. Bu sayede üretilen her nokta, görsel konteks ile desteklenmiş olur
preprints.org
.
Oto-regresif Şerit Çıkarımı (Ardışıl Tahmin)
Transformer decoder, şerit noktalarını oto-regresif olarak çıkarır, yani bir seferde bir token (noktayı) üreterek sırayla ilerler
preprints.org
. Başlangıçta her şerit için bir başlangıç bilgisi (örn. başlangıç noktası veya özel bir <START> tokenı) verilir. Sonra model ardışık olarak her adımda bir sonraki noktayı tahmin eder ve bu tahmini bir sonraki adıma giriş olarak besler. Bu süreç, bir <EOS> (dizi sonu) tokenı üretilene veya maksimum uzunluğa ulaşılana kadar devam eder
preprints.org
. Aşağıdaki kod, bir şeridin nokta dizisini oto-regresif üretmeyi basitleştirerek gösteriyor:
# Varsayalım encoder çıktılarını (visual_tokens) elde ettik
visual_tokens = ...  # [M, 1, E] boyutlu görsel bellek (M token, tek resim için)
decoder = TransformerDecoder(...)  # birden çok DecoderLayer içeren decoder
linear_head = nn.Linear(256, vocab_size)  # tokenları id'lere eşleyen çıkış katmanı

lane_sequence = []
input_tokens = [START_TOKEN_ID]  # başlangıç tokenı
for step in range(max_len):
    tgt_emb = token_embedding(input_tokens)        # token id'lerini embedding'e çevir
    tgt_emb = pos_encoding(tgt_emb)               # konum bilgisi ekle
    output_emb = decoder(tgt_emb.transpose(0,1),   # shape: [T, B, E] transpoze ile (T adım, batch=1)
                         visual_tokens)            # görsel bellek ile decode et
    pred_logits = linear_head(output_emb[-1])     # son adımdaki çıktı için tahminler
    pred_token = torch.argmax(pred_logits, dim=-1).item()  # en olası tokenı seç
    if pred_token == EOS_TOKEN_ID:
        break  # dizinin sonu
    lane_sequence.append(pred_token)
    input_tokens.append(pred_token)  # yeni tokenı girdiye ekle, döngüye devam
Yukarıda, TransformerDecoder birden fazla decoder katmanını içeren tüm kod çözücüyü temsil ediyor. Her adımda mevcut şerit dizi embedding’ine görsel bellek ile cross-attention uygulayarak bir sonraki tokenı üretiyoruz. linear_head, decoder çıktısını gerçek bir token kimliğine (örneğin bir kordinat quantization indeksi) dönüştüren basit bir doğrusal katmandır. Üretilen token her seferinde dizinin sonuna eklenerek bir sonraki adıma beslenir. Model, bir <EOS> (End of Sequence) tokenı ürettiğinde durur
preprints.org
. Neden gerekli: Bu ardışıl üretim yaklaşımı, şerit noktalarının değişken uzunlukta ve sıralı bir biçimde çıkarılmasını sağlar. Her yeni nokta, önceki noktalara ve görüntüye dayanarak üretildiği için, ortaya çıkan şerit geometrik olarak tutarlı olur.
Mühendis Gözüyle Değerlendirme
Bu LaneLM benzeri mimari, yapısal olarak mantıklı bir tasarım sunuyor. CNN tabanlı görsel encoder ile güçlü görsel özellikler çıkarılırken, Transformer decoder bu özellikleri kullanarak şeritleri bir dizi olarak modelleyebiliyor. Bu sayede model, tıpkı bir cümle yazar gibi şerit noktalarını sırayla üretebiliyor ve her noktada görüntünün ilgili bölümlerine bakabiliyor – bu oldukça yenilikçi ve etkilidir. Mimari birkaç farklı parça içerdiği için karmaşık görünebilir, ancak her bileşenin belirli bir amacı var ve birbirini tamamlıyor. Örneğin, CNN’in çıkaramadığı dizisel bağıntıları Transformer üstleniyor, Transformer’ın ham veride zorlanacağı görsel ayrıntıları ise CNN sağlıyor. Sonuç olarak, mimari mantıklı bir şekilde tasarlanmış ve özellikle zorlu şerit tespit senaryolarında güçlü bir performans hedefliyor. Yine de, klasik basit yöntemlere kıyasla daha fazla bileşen barındırdığı için uygulaması ve eğitimi daha zahmetli olabilir. Ancak elde edilen esneklik ve doğruluk artışı göz önüne alındığında, bu ekstra karmaşıklık gereksiz değil, amaca hizmet eden bir tasarım olarak değerlendirilebilir.
preprints.org
preprints.org
Alıntılar

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

LaneLM: Lane Detection as Language Modeling[v1] | Preprints.org

https://www.preprints.org/manuscript/202504.1582

Elbette. LaneLM makalesindeki transformer tabanlı şerit tespit modelini sıfırdan uygulayabilmen için, benzer açık kaynaklı çalışmaları ve bu çalışmalarda kullanabileceğin kod kısımlarını araştıracağım. Özellikle LaneATT, LSTR, LaneFormer, PersFormer gibi projelerde bulunan ilgili transformer, cross-attention veya decoder yapılarını örnek olarak göstereceğim. Böylece doğrudan bu projelere bakarak kendi modelini oluşturabileceksin.

Araştırmam tamamlandığında sana hangi projeden hangi yapıları alabileceğini açık şekilde göstereceğim.

Transformer Tabanlı Lane Detection Örnekleri
1. LSTR (Lane Shape Prediction with Transformers)
Açıklama: LSTR, bir CNN tabanlı görsel backbone üzerine transformer bloğu entegre ederek doğrudan şerit eğrisi parametrelerini çıkartan bir modeldir
ar5iv.labs.arxiv.org
. Transformer katmanı, kendine-özgü dikkat (self-attention) mekanizmasıyla görüntüdeki uzun ve ince şerit yapıları ile küresel bağlam bilgisini yakalar
ar5iv.labs.arxiv.org
. Model, çıktıda her bir şeridi bir polinom parametre vektörü olarak verir ve bu çıktıları Hungarian eşleştirmesi ile gerçek veriye atayarak eğitilir. Bu tek-aşamalı yaklaşım, öngörülen şeritlerle gerçek şeritleri bire bir eşleştirerek NMS (non-max suppression) ihtiyacını ortadan kaldırır
ar5iv.labs.arxiv.org
.
LaneLM’e Faydası: LSTR’nin encoder-decoder benzeri yapısı, özellikle transformer decoder’ının görüntü özelliklerine cross-attention ile bakarak şerit parametreleri üretmesi, LaneLM’i sıfırdan uygularken yararlı olacaktır. Ayrıca Hungarian eşleştirmeli biricik (one-to-one) atama stratejisi ve uçtan uca eğitim mantığı, birden fazla şerit çıktısını yönetme konusunda LaneLM’e ilham verebilir.
GitHub: liuruijin17/LSTR (PyTorch uygulaması, resmi kaynak kodu)
ar5iv.labs.arxiv.org
.
2. O2SFormer (One-to-Several Transformer)
Açıklama: O2SFormer, DETR tarzı (end-to-end transformer tabanlı) bir şerit tespit mimarisidir. Klasik DETR’nin tek-eşleme (one-to-one) etiket atama kısıtını aşmak için one-to-several adı verilen hibrit bir atama stratejisi önerir
github.com
. Bu sayede her gerçek şerit için birden çok sorgu (query) eşleşmesine izin vererek eğitimi hızlandırır, ancak yine de uçtan uca bir yapıyı korur
github.com
. Ayrıca O2SFormer, transformer decoder aşamasında dinamik anchor tabanlı konum sorguları kullanır; yani önceden tanımlanmış şerit anchor’ları üzerinden konumsal gömüler oluşturarak sorgulara belirgin uzamsal önbilgi katar
github.com
. Katman bazlı yumuşak etiketleme gibi yeniliklerle, O2SFormer CULane gibi veri setlerinde hem transformer tabanlı hem de CNN tabanlı önceki yöntemleri geçmektedir
github.com
.
LaneLM’e Faydası: Bu projenin açık kaynak kodu, DETR benzeri bir transformer decoder’ın nasıl uygulandığını gösterdiği için değerlidir. Özellikle cross-attention ile görüntü özelliklerine bakan sorguların oluşturulması, anchor destekli konumsel gömme işlemleri ve Hungarian benzeri eşleme mantığı gibi konular, LaneLM tarzı bir model geliştirirken doğrudan fayda sağlayacaktır.
GitHub: zkyseu/O2SFormer (PyTorch uygulaması, mmdetection tabanlı).
3. CondLSTR (Dynamic Kernels via Transformers)
Açıklama: “Generating Dynamic Kernels via Transformers for Lane Detection” çalışması (CondLSTR), transformer yapısını kullanarak her bir şerit için ayrı bir evrişimsel çekirdek (kernel) üreten yenilikçi bir yaklaşımdır
github.com
. Bir backbone ağından çıkan özellik haritası üzerinde, transformer decoder’ı her şerit çizgisine karşılık gelen dinamik konvolüsyon filtrelerini hesaplar; ardından bu filtreler ilgili şeridi özelleşmiş olarak tespit etmek için görüntü özelliğine uygulanır
github.com
. Bu sayede model, öğrenilmiş sorgular aracılığıyla her şeride özgü bir algılayıcı oluşturmuş olur.
LaneLM’e Faydası: CondLSTR’nin kod yapısı, bir transformer çıktısının nasıl downstream bir işleme (ör. dinamik evrişim) dönüştürülebileceğine dair önemli bir örnek sunuyor. LaneLM mimarisi kurgulanırken, bu projedeki cross-attention kullanımının şerit bazlı özellik çıkarma veya şerit belirteci (token) üretme için nasıl entegre edildiği incelenebilir. Özellikle görsel özellikten gelen token’ların transformer ile işlenip belirli bir göreve (şerit maskesi oluşturma gibi) yönlendirilmesi, LaneLM’de decoder tasarımı açısından yol gösterici olacaktır.
GitHub: czyczyyzc/CondLSTR (PyTorch kodları, dinamik konvolüsyon yaklaşımlı).
4. LATR (Lane detection with TRansformer, 3D)
Açıklama: LATR, tek bir kameradan 3B şerit tespiti için tasarlanmış bir transformer mimarisidir. Bu model, görüntüden kuşbakışı dönüşümler oluşturmadan, doğrudan ön görüş özellikleri üzerinden sorgu-key/value tabanlı cross-attention ile 3B şeritleri çıkarır
arxiv.org
. LATR’da transformer decoder sorguları, görüntüden çıkarılan 2B şerit özelliklerine dayalı olarak başlatılır ve her bir sorguya güncellenen bir 3B zemin düzlemi üzerinden hesaplanan dinamik 3B konumsal gömme eklenir
arxiv.org
. Bu sayede her sorgu, hem görüntü içeriğini hem de uzamsal geometrik bilgiyi bünyesinde barındırarak, 3B uzayda şerit tahmini yapar. LATR, Apollo (sanal) ve OpenLane gibi gerçek veri setlerinde önceki yöntemlere kıyasla anlamlı performans artışları raporlamıştır.
LaneLM’e Faydası: LATR’ın açık kaynak kodu, transformer tabanlı bir şerit tespit algoritmasının uçtan uca nasıl kurulabileceğini gösterir. Özellikle cross-attention mekanizması ile sorguların görsel özellikleri kullanarak nasıl şerit çıktıları ürettiği ve sorguların başlangıçta nasıl tanımlandığı (örneğin, görüntü özelliklerinden türetilen şerit ipuçlarıyla) gibi konular, LaneLM benzeri bir model geliştirirken oldukça değerlidir. 3B şerit tespitine odaklansa da, LATR’daki görsel encoder + transformer decoder yapısı ve kod altyapısı 2B şerit modeli için büyük ölçüde uyarlanabilir.
GitHub: JMoonr/LATR (PyTorch kodları, ICCV 2023 resmi uygulaması).
Alıntılar

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

[2011.04233] End-to-end Lane Shape Prediction with Transformers

https://ar5iv.labs.arxiv.org/html/2011.04233

GitHub - zkyseu/O2SFormer: End-to-End Lane detection with One to Several Transformer

https://github.com/zkyseu/O2SFormer

GitHub - zkyseu/O2SFormer: End-to-End Lane detection with One to Several Transformer

https://github.com/zkyseu/O2SFormer

GitHub - zkyseu/O2SFormer: End-to-End Lane detection with One to Several Transformer

https://github.com/zkyseu/O2SFormer

GitHub - czyczyyzc/CondLSTR: Code for paper "Generating Dynamic Kernels via Transformers for Lane Detection"

https://github.com/czyczyyzc/CondLSTR

[2308.04583] LATR: 3D Lane Detection from Monocular Images with Transformer

https://arxiv.org/abs/2308.04583

