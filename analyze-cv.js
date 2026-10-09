import { GoogleGenAI, Type } from "@google/genai";
import "dotenv/config";
import * as readline from "readline";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = (query) => {
  return new Promise((resolve) => rl.question(query, resolve));
};

// CV Şeması (Yapısı Netleştirilmiş)
const cvResponseSchema = {
  type: Type.OBJECT,
  properties: {
    personalInfo: {
      type: Type.OBJECT,
      properties: {
        fullName: { type: Type.STRING },
        title: { type: Type.STRING },
        location: { type: Type.STRING },
        email: { type: Type.STRING },
        phone: { type: Type.STRING },
        githubUrl: {
          type: Type.STRING,
          description: "Kullanıcının paylaştığı GitHub profil adresi",
        },
        linkedinUrl: {
          type: Type.STRING,
          description: "Kullanıcının paylaştığı LinkedIn profil adresi",
        },
      },
    },
    education: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          institution: { type: Type.STRING },
          degree: { type: Type.STRING },
          startYear: { type: Type.STRING },
          endYear: { type: Type.STRING },
        },
        required: ["institution", "degree"],
      },
    },
    workExperience: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          company: { type: Type.STRING },
          role: { type: Type.STRING },
          employmentType: { type: Type.STRING },
          location: { type: Type.STRING },
          startDate: { type: Type.STRING },
          endDate: { type: Type.STRING },
          techStack: { type: Type.ARRAY, items: { type: Type.STRING } },
          description: { type: Type.STRING },
        },
        required: ["company", "role"],
      },
    },
    projects: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          name: { type: Type.STRING },
          platform: { type: Type.STRING },
          techStack: { type: Type.ARRAY, items: { type: Type.STRING } },
          status: { type: Type.STRING },
          description: { type: Type.STRING },
          publicUrl: {
            type: Type.STRING,
            description:
              "Projenin App Store, Google Play Store, Web veya GitHub bağlantısı",
          },
        },
        required: ["name"],
      },
    },
    skillsWithProficiency: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          skill: { type: Type.STRING },
          category: { type: Type.STRING },
          level: { type: Type.STRING },
        },
        required: ["skill", "level"],
      },
    },
    smartVerificationPrompts: {
      type: Type.ARRAY,
      description: "Metinden çıkarılan doğrulama soruları",
      items: {
        type: Type.OBJECT,
        properties: {
          targetItem: { type: Type.STRING },
          type: { type: Type.STRING },
          reason: { type: Type.STRING },
          question: { type: Type.STRING },
        },
        required: ["targetItem", "type", "question"],
      },
    },
  },
  required: [
    "education",
    "workExperience",
    "projects",
    "skillsWithProficiency",
    "smartVerificationPrompts",
  ],
};

async function analyzeInitialText(rawBioText) {
  const systemInstruction = `
    Sen yazılımcı özgeçmişleri (CV) üzerinde uzmanlaşmış, teknik yetkinlik ve proje analiz asistansın.
    Görevin, kullanıcının serbest metin olarak sunduğu bilgileri analiz etmek, ham verileri JSON şemasına dönüştürmek ve CV'yi zenginleştirmek adına nokta atışı doğrulama/tamamlama soruları üretmektir.

    1. EKSİKSİZ METİN ANALİZİ: Eğitim, iş tecrübesi, konum ve teknolojileri ayıkla.
    2. AKILLI LİNK TESPİTİ (LINK_REQUEST): AppStore, PlayStore, GitHub'da yayınlandığı belirtilen ancak URL'si verilmeyen projeler için 'LINK_REQUEST' türünde soru ekle. GitHub profili için de sor.
    3. TEKNOLOJİ TEYİDİ (TECH_PROFICIENCY): Ekosistemlerden çıkarım yapıp teyit sorusu üret.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3.8-flash",
    contents: rawBioText,
    config: {
      systemInstruction,
      responseMimeType: "application/json",
      responseSchema: cvResponseSchema,
      temperature: 0.2,
    },
  });

  return JSON.parse(response.text);
}

async function mergeUserAnswers(currentJson, answers) {
  const systemInstruction = `
    Sen gelen kullanıcı yanıtlarını CV verisine kusursuz bir şekilde eşleyen bir JSON birleştirme asistanısın.

    KESİN KURALLAR:
    1. PROJE LİNKLERİ: Kullanıcının cevaplarında sunduğu her bir 'http' veya 'https' ile başlayan bağlantıyı (URL), 'targetItem' alanındaki proje adıyla eşleştir ve ilgili projenin 'projects[].publicUrl' alanına YAZ.
       - Örnek: targetItem "CardEng" ve cevap "https://play.google.com/..." ise -> 'CardEng' projesinin 'publicUrl' değerine bu URL'yi koy.
       - Örnek: targetItem "Akıllı Abonelik" ve cevap "https://apps.apple.com/..." ise -> 'Akıllı Abonelik' projesinin 'publicUrl' değerine koy.
       - Örnek: targetItem "Canlı FX Cüzdanım" ve cevap "https://apps.apple.com/..." ise -> 'Canlı FX Cüzdanım' projesinin 'publicUrl' değerine koy.

    2. GITHUB / SOSYAL LİNKLER:
       - targetItem "GitHub" ise ve gelen cevap bir URL ise, bunu 'personalInfo.githubUrl' alanına yaz.

    3. TEKNOLOJİ YETKİNLİKLERİ:
       - Kullanıcının onayladığı yetkinlik ve seviye bilgilerini (Başlangıç/Orta/İleri - Beginner/Intermediate/Advanced) 'skillsWithProficiency' dizisine ekle veya var olanı güncelle.

    4. 'smartVerificationPrompts' dizisini en son boş bir dizi ([]) yap veya sil.
  `;

  const prompt = `
    MEVCUT CV JSON:
    ${JSON.stringify(currentJson, null, 2)}

    KULLANICININ VERDİĞİ CEVAPLAR VE LİNKLER:
    ${JSON.stringify(answers, null, 2)}
  `;

  const response = await ai.models.generateContent({
    model: "gemini-3.8-flash",
    contents: prompt,
    config: {
      systemInstruction,
      responseMimeType: "application/json",
      responseSchema: cvResponseSchema,
      temperature: 0.1,
    },
  });

  return JSON.parse(response.text);
}

async function runConsoleApp() {
  console.log("\n===========================================");
  console.log("  AI TAHMİN & LİNK DOĞRULAMA SİSTEMİ      ");
  console.log("===========================================\n");

  const sampleBio = `2020 yılında giriş yaptığım Çukurova üniversitesi bilgisayar mühendisliğinde 2022 yılından itibaren yazılım geliştirme projerlerinde yer aldım. 2022 yazında mersinde bulunan Kardelen yazılımda ilk full stack projemen olan insan kaynakları yönetim sistemini React ve Nodejs ile stajımı tamamladım. 3. sınıfta asewise firmasınının teklif sisteminin backend ve frontend kısmını tamamladım. Bu süreçte aynı zamanda mobil geliştirme alanındaki ilk projem olan CardEng ingilizce kelime öğrenme uygulamasını geliştirdim ve Google play üzerinde yayınlamasını yaptım. 4.sınıfta part-time olarak öğrencilere yönelik uygulama olan Tovivu da React frontend geliştirici olarak görev yaptım. Mezun olmadan 1 ay öncesinde Adanada bulunan ve e-dönüşüm hizmetleri veren Hızlı bilişim şirketine full stack geliştirci olarak girdim ve yaklaşık 1.5 yıldır orada full time çalışmaktayım. GIB'in belirlediği geliştirmelerin web ve masaüstü (DevExpress) üzerinden entegrasyonlarını sağladım. Net ve angular teknolojileri kullandım mysql, restApi, SOAP gibi backend teknolojilerinde çalıştım. Eş zamanlı olarak Payment firması olan Payport da frontend geliştirici olarak uçta uca CRM ve MYS projesini geliştirdim. Bu süreçte Jira, Docker, Jenkis gibi teknolojileri de kullandım ve geliştirmelerine devam etmekteyim. İşten kalan zamanlarda geliştirdiğim ve React Native Expo ile Akıllı Abonellik ve Canlı FX Cüzdanım uygulamalarını AppStore' da yayına aldım.`;

  console.log("[1/3] Metin analiz ediliyor...");
  let cvData = await analyzeInitialText(sampleBio);

  const prompts = cvData.smartVerificationPrompts || [];

  if (prompts.length > 0) {
    console.log(
      `\n[2/3] Toplam ${prompts.length} adet dinamik soru tespit edildi:\n`,
    );
    const userAnswers = [];

    for (let i = 0; i < prompts.length; i++) {
      const item = prompts[i];
      const tag =
        item.type === "LINK_REQUEST"
          ? "\x1b[33m[Link İsteme]\x1b[0m"
          : "\x1b[36m[Teknoloji Teyidi]\x1b[0m";

      console.log(`${tag} Hedef: ${item.targetItem}`);
      const answer = await askQuestion(
        `Soru ${i + 1}: ${item.question}\nCevap: `,
      );

      userAnswers.push({
        targetItem: item.targetItem,
        type: item.type,
        question: item.question,
        userAnswer: answer,
      });
      console.log("");
    }

    console.log("[3/3] Yanıtlar ve linkler CV verisine işleniyor...");
    cvData = await mergeUserAnswers(cvData, userAnswers);
  }

  delete cvData.smartVerificationPrompts;

  console.log("\n===========================================");
  console.log("       LİNK VE YETKİNLİK EKLENMİŞ NİHAİ JSON ");
  console.log("===========================================\n");
  console.log(JSON.stringify(cvData, null, 2));

  rl.close();
}

runConsoleApp().catch((err) => {
  console.error("Hata:", err);
  rl.close();
});
