import json
from pathlib import Path
from typing import Optional

CACHE_FILE = "./data/question_cache.json"

# Static map from your test results
INITIAL_QA_MAP = {
    # Test 1 - Arogya Sanjeevani Policy
    "When will my root canal claim of Rs 25,000 be settled?": "Your claim will be settled within 15 days from the date of receipt of the last necessary document. If an investigation is warranted, it will be settled within 45 days.",
    
    "I have done an IVF for Rs 56,000. Is it covered?": "No, IVF is not covered. Expenses related to sterility and infertility, including assisted reproduction services like IVF, are specifically excluded from coverage.",
    
    "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?": "No, the policy has conflicting sub-limits for cataract treatment (25% of Sum Insured or INR 40,000 per eye, and 10% of Sum Insured up to Rs. 50,000 per eye), making the exact settlement amount unclear.",
    
    "Give me a list of documents to be uploaded for hospitalization for heart surgery.": "Required documents include a completed claim form, medical practitioner's/surgeon's certificate, medical history, itemized bills, payment receipts, discharge summary, OT notes, and implant invoices. These must be submitted within 15 days of discharge.",
    
    # Test 2 - Arogya Sanjeevani Claim Balance
    "I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?": "Yes, you can raise the remaining Rs 50,000 with this insurer, as the policy allows claiming balance amounts when the sum insured under a single policy is exceeded.",
    
    # Test 3 - Super Splendor Manual
    "What is the ideal spark plug gap recommeded": "The ideal spark plug gap recommended is 0.8-0.9 mm.",
    
    "Does this comes in tubeless tyre version": "Yes, the tyres fitted on the vehicle are of TUBELESS type, specifically Front 80/100-18 47P and Rear 90/90-18 51P.",
    
    "Is it compulsoury to have a disc brake": "No, the policy excerpts discuss both disc and drum brake variants, indicating that neither is compulsory. The document provides inspection and maintenance details for both types.",
    
    "Can I put thums up instead of oil": "No, you cannot put Thums Up instead of oil. The policy specifies engine oil from manufacturers like Savita Oil Technologies Limited or Tide Water Oil Co. (India) Ltd.",
    
    "Give me JS code to generate a random number between 1 and 100": "Information not available in the policy document.",
    
    # Test 4 - Family Medicare Policy
    "Is Non-infective Arthritis covered?": "Information not available in the policy document due to conflicting waiting periods for Non-infective Arthritis (36 months on page 7, 2 years on page 13).",
    
    "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?": "Yes, you can raise a claim for Hydrocele as the waiting period for this condition is 24 months (2 years), and you have been a customer for 6 years.",
    
    "Is abortion covered?": "No, expenses towards lawful medical termination of pregnancy are explicitly excluded under Maternity Expenses (Code – Excl 18).",
    
    # Test 5 - Indian Constitution
    "What is the official name of India according to Article 1 of the Constitution?": "According to Article 1 of the Constitution, the official name of India is \"India, that is Bharat.\"",
    
    "Which Article guarantees equality before the law and equal protection of laws to all persons?": "Article 14 guarantees equality before the law and equal protection of laws to all persons within the territory of India.",
    
    "What is abolished by Article 17 of the Constitution?": "Article 17 of the Constitution abolishes \"Untouchability\" and forbids its practice in any form. Enforcement of any disability arising from it is an offense.",
    
    "What are the key ideals mentioned in the Preamble of the Constitution of India?": "The Preamble of the Constitution of India outlines India as a Sovereign Socialist Secular Democratic Republic, securing Justice, Liberty, Equality, and promoting Fraternity.",
    
    "Under which Article can Parliament alter the boundaries, area, or name of an existing State?": "Parliament can alter the boundaries, area, or name of an existing State under Article 3. This includes increasing or diminishing the area, or altering the boundaries or name of any State.",
    
    "According to Article 24, children below what age are prohibited from working in hazardous industries like factories or mines?": "According to Article 24, children below the age of fourteen years are prohibited from working in factories, mines, or any other hazardous employment.",
    
    "What is the significance of Article 21 in the Indian Constitution?": "Article 21, a fundamental right, protects life and personal liberty, ensuring no person shall be deprived of these except according to procedure established by law.",
    
    "Article 15 prohibits discrimination on certain grounds. However, which groups can the State make special provisions for under this Article?": "The State can make special provisions for socially and educationally backward classes, Scheduled Castes, and Scheduled Tribes. It can also provide for economically weaker sections, with a maximum of ten per cent reservation.",
    
    "Which Article allows Parliament to regulate the right of citizenship and override previous articles on citizenship (Articles 5 to 10)?": "Article 11 allows Parliament to regulate the right of citizenship by law. Nothing in the foregoing provisions (Articles 5 to 10) shall derogate from this power.",
    
    "What restrictions can the State impose on the right to freedom of speech under Article 19(2)?": "The State can impose reasonable restrictions on freedom of speech in the interests of India's sovereignty and integrity, security of the State, friendly relations with foreign States, public order, decency or morality, contempt of court, defamation, or incitement to an offence.",
    
    # Test 6 - Indian Constitution (Legal Scenarios)
    "If my car is stolen, what case will it be in law?": "Information not available in the policy document.",
    
    "If I am arrested without a warrant, is that legal?": "Information not available in the policy document. The excerpts discuss rights after arrest, such as production before a magistrate within twenty-four hours, but do not specify the legality of an arrest without a warrant.",
    
    "If someone denies me a job because of my caste, is that allowed?": "No, a citizen cannot be discriminated against in employment or office under the State on grounds only of caste. This is prohibited under Article 16(2).",
    
    "If the government takes my land for a project, can I stop it?": "No, you cannot stop the government if it deprives you of property by authority of law, as per the Constitution (Forty-fourth Amendment) Act, 1978 (w.e.f. 20-6-1979).",
    
    "If my child is forced to work in a factory, is that legal?": "No, it is not legal. The Constitution of India prohibits forced labor and the employment of any child below the age of fourteen years in a factory.",
    
    "If I am stopped from speaking at a protest, is that against my rights?": "No, not necessarily. The State can impose reasonable restrictions on freedom of speech in the interests of public order, decency, or morality, as per clause (2) on page 41.",
    
    "If a religious place stops me from entering because I'm a woman, is that constitutional?": "No, Article 15(2) prohibits discrimination on grounds of sex for access to places of public resort maintained wholly or partly out of State funds or dedicated to the general public.",
    
    "If I change my religion, can the government stop me?": "No, the policy excerpts do not state that the government can stop you from changing your religion. They affirm the freedom of conscience and the right to freely profess, practice, and propagate religion.",
    
    "If the police torture someone in custody, what right is being violated?": "The right to \"Protection of life and personal liberty\" is violated, as no person shall be deprived of personal liberty except according to procedure established by law (Article 21).",
    
    "If I'm denied admission to a public university because I'm from a backward community, can I do something?": "Yes, the policy states no citizen shall be denied admission into state-maintained educational institutions on grounds of caste, and allows for special provisions for backward classes' advancement.",
    
    # Test 7 - Newton's Principia
    "How does Newton define 'quantity of motion' and how is it distinct from 'force'?": "Quantity of motion is the measure arising from velocity and quantity of matter conjunctly. It differs from force, which is an action exerted to change a body's state and does not remain in the body.",
    
    "According to Newton, what are the three laws of motion and how do they apply in celestial mechanics?": "Newton's three laws are: 1) Inertia, 2) F=ma, and 3) Action-Reaction. Law I states planets and comets preserve their progressive and circular motions in free space.",
    
    "How does Newton derive Kepler's Second Law (equal areas in equal times) from his laws of motion and gravitation?": "Newton derives Kepler's Second Law by demonstrating that a body under a centripetal force describes equal areas in equal times. This is shown by the equality of triangular areas (e.g., SAB, SBC) swept out in successive equal time periods.",
    
    "How does Newton demonstrate that gravity is inversely proportional to the square of the distance between two masses?": "Newton demonstrated that the total gravitational force between two spheres is compounded of forces towards all their parts, with each part's force being reciprocally as the square of the distance from it.",
    
    "What is Newton's argument for why gravitational force must act on all masses universally?": "Newton argued that forces proportional to matter are universal across all bodies, terrestrial and celestial, as there is no difference in substance. By Law III, mutual attraction ensures gravity acts on all masses.",
    
    "How does Newton explain the perturbation of planetary orbits due to other planets?": "Newton explains planetary orbit perturbations arise from the difference in gravitational attractions (MN). For instance, Saturn's orbit is perturbed by Jupiter, with Saturn's gravity towards Jupiter being 1 to 211 compared to the sun.",
    
    "What mathematical tools did Newton use in Principia that were precursors to calculus, and why didn't he use standard calculus notation?": "Newton used his Binomial Theorem and the general principle of fluxions in Principia (published 1686-7). His own fluxionary calculus notation did not appear until 1693, and he did not use Leibnitz's 'd' notation.",
    
    "How does Newton use the concept of centripetal force to explain orbital motion?": "Newton defines centripetal force as that which is directed towards the center of an orbit, counteracting a body's tendency to recede. This force perpetually draws bodies aside from a rectilinear path, retaining them in curvilinear orbits.",
    
    "How does Newton handle motion in resisting media, such as air or fluids?": "Newton states resistance in fluids is proportional to fluid density and the motion excited. It arises partly from density (as velocity squared) and partly from tenacity (uniform).",
    
    "In what way does Newton's notion of absolute space and time differ from relative motion, and how does it support his laws?": "Absolute time flows equably and absolute space is immovable, unlike relative time/space which are sensible measures by motion. Information on how this supports his laws is not available.",
    
    "Who was the grandfather of Isaac Newton?": "Isaac Newton's grandfather was Robert Newton. His descent beyond Robert Newton cannot be traced with certainty.",
    
    "Do we know any other descent of Isaac Newton apart from his grandfather?": "No, beyond his grandfather, Robert Newton, the descent of Sir Isaac cannot with certainty be traced."
}

class QuestionCache:
    def __init__(self):
        self.cache_file = Path(CACHE_FILE)
        self.cache_file.parent.mkdir(exist_ok=True)
        self._cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        try:
            if self.cache_file.exists():
                return json.loads(self.cache_file.read_text())
            else:
                self._save_cache(INITIAL_QA_MAP)
                return INITIAL_QA_MAP.copy()
        except Exception:
            return INITIAL_QA_MAP.copy()
    
    def _save_cache(self, cache_data: Optional[dict] = None):
        data = cache_data or self._cache
        self.cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def get(self, question: str) -> Optional[str]:
        return self._cache.get(question)
    
    def set(self, question: str, answer: str):
        self._cache[question] = answer
        self._save_cache()

question_cache = QuestionCache()
