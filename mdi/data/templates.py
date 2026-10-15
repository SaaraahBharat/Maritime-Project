"""Caller phrase banks per language.

Each slot has several alternatives. The LAST alternative of every slot is held
out: train never sees it, so test scores on `unseen_phrasing` examples measure
generalisation instead of template memorisation.

Hindi/Tamil/Malayalam phrases deliberately keep English protocol words
(MAYDAY, MMSI, over, compass points) because that code-switching is how VHF
traffic on the Indian coast actually sounds. They were written by a non-native
author and SHOULD be reviewed by a native speaker before the report is final.

Placeholders: {CALL} MAYDAY/PAN-PAN, {V} vessel, {M} MMSI, {T} vessel type,
{P} position, {N} persons on board, {L} landmark, {D} distance, {C} compass.
"""

from __future__ import annotations

from mdi.schema import IncidentType, Language, VesselType

I = IncidentType

PHRASES: dict[Language, dict[str, list[str]]] = {
    Language.EN: {
        "opening": ["{CALL} {CALL} {CALL}, this is {V} {V} {V}", "{CALL}, this is {V}", "{CALL} {CALL}, {V} calling"],
        "opening_noname": ["{CALL} {CALL} {CALL}, anybody hear me", "{CALL}, please respond", "{CALL} {CALL}, coast guard come in"],
        "mmsi": ["MMSI {M}", "my MMSI is {M}", "MMSI number {M}"],
        "vessel_type": ["we are a {T}", "this is a {T}", "{T}"],
        "position": ["my position {P}", "position {P}", "we are {P}"],
        "relative": ["{D} miles {C} of {L}", "{D} nautical miles {C} of {L}", "about {D} miles {C} from {L}"],
        "pob": ["{N} persons on board", "we have {N} people on board", "{N} crew on board"],
        "assist": ["require immediate assistance", "need help urgently", "request immediate assistance"],
        "close": ["over", "over", ""],
        "say_again": ["say again, say again", "cannot hear you", "repeat please"],
        I.FIRE: ["we have a fire on board", "fire in the engine room", "vessel on fire"],
        I.FLOODING: ["we are taking water", "flooding in the hold", "we are sinking, water coming in"],
        I.ENGINE_FAILURE: ["engine failure", "engine broken down, we are drifting", "engine not working"],
        I.CAPSIZING: ["we are capsizing", "vessel has capsized", "boat is listing heavily"],
        I.MAN_OVERBOARD: ["man overboard", "one person fell in the water", "crew member lost overboard"],
        I.MEDICAL: ["one crew seriously ill", "one person badly injured", "crew member unconscious"],
        I.COLLISION: ["we have collided with another vessel", "a ship hit our boat", "collision"],
        I.GROUNDING: ["we are aground on rocks", "vessel aground on sand bank", "we hit the bottom and are stuck"],
    },
    Language.HI: {
        "opening": ["{CALL} {CALL} {CALL}, यह {V} है, {V}", "{CALL}, हम {V} से बोल रहे हैं", "{CALL} {CALL}, {V} बोल रहा है"],
        "opening_noname": ["{CALL} {CALL}, कोई सुन रहा है", "{CALL}, कोस्ट गार्ड जवाब दीजिए", "{CALL} {CALL}, मदद चाहिए"],
        "mmsi": ["MMSI {M}", "हमारा MMSI {M} है", "MMSI नंबर {M}"],
        "vessel_type": ["हम {T} हैं", "यह एक {T} है", "{T} है"],
        "position": ["हमारी पोज़ीशन {P}", "पोज़ीशन {P}", "हमारी जगह {P}"],
        "relative": ["{L} से {D} मील {C}", "{L} से करीब {D} मील {C}", "{L} से {D} नॉटिकल मील {C}"],
        "pob": ["बोट पर {N} लोग हैं", "हम {N} लोग हैं", "{N} आदमी सवार हैं"],
        "assist": ["तुरंत मदद चाहिए", "जल्दी मदद भेजिए", "हमें तुरंत सहायता चाहिए"],
        "close": ["ओवर", "over", ""],
        "say_again": ["सुनाई नहीं दे रहा, फिर से बोलिए", "क्या? फिर से बोलिए", "आवाज़ नहीं आ रही"],
        I.FIRE: ["बोट में आग लगी है", "इंजन रूम में आग लगी है", "जहाज़ पर आग लगी है"],
        I.FLOODING: ["बोट में पानी भर रहा है", "नीचे से पानी आ रहा है", "हम डूब रहे हैं, पानी भर रहा है"],
        I.ENGINE_FAILURE: ["इंजन बंद हो गया है", "इंजन खराब हो गया, हम बह रहे हैं", "इंजन काम नहीं कर रहा"],
        I.CAPSIZING: ["बोट पलट रही है", "नाव पलट गई है", "बोट एक तरफ़ झुक रही है"],
        I.MAN_OVERBOARD: ["एक आदमी पानी में गिर गया है", "एक आदमी समुद्र में गिर गया", "हमारा एक साथी पानी में गिर गया"],
        I.MEDICAL: ["एक आदमी बहुत बीमार है", "एक आदमी को गंभीर चोट लगी है", "एक साथी बेहोश है"],
        I.COLLISION: ["दूसरे जहाज़ से टक्कर हो गई", "हमारी बोट किसी जहाज़ से टकरा गई", "टक्कर हो गई है"],
        I.GROUNDING: ["बोट चट्टान पर फँस गई है", "हम रेत में फँस गए हैं", "बोट नीचे से टकराकर फँस गई"],
    },
    Language.TA: {
        "opening": ["{CALL} {CALL} {CALL}, இது {V}, {V}", "{CALL}, நாங்கள் {V} படகிலிருந்து பேசுகிறோம்", "{CALL} {CALL}, {V} பேசுகிறது"],
        "opening_noname": ["{CALL} {CALL}, யாராவது கேட்கிறீர்களா", "{CALL}, கோஸ்ட் கார்டு பதில் சொல்லுங்கள்", "{CALL} {CALL}, உதவி வேண்டும்"],
        "mmsi": ["MMSI {M}", "எங்கள் MMSI {M}", "MMSI எண் {M}"],
        "vessel_type": ["நாங்கள் {T}", "இது ஒரு {T}", "{T}"],
        "position": ["எங்கள் பொசிஷன் {P}", "பொசிஷன் {P}", "நாங்கள் இருக்கும் இடம் {P}"],
        "relative": ["{L}-இலிருந்து {D} மைல் {C}", "{L}-இலிருந்து சுமார் {D} மைல் {C}", "{L}-க்கு {D} நாட்டிகல் மைல் {C}"],
        "pob": ["படகில் {N} பேர் இருக்கிறோம்", "நாங்கள் {N} பேர்", "{N} பேர் இருக்கிறார்கள்"],
        "assist": ["உடனடி உதவி தேவை", "சீக்கிரம் உதவி அனுப்புங்கள்", "எங்களுக்கு உடனே உதவி வேண்டும்"],
        "close": ["ஓவர்", "over", ""],
        "say_again": ["கேட்கவில்லை, மீண்டும் சொல்லுங்கள்", "என்ன? திரும்பச் சொல்லுங்கள்", "சத்தம் கேட்கவில்லை"],
        I.FIRE: ["படகில் தீ பிடித்துவிட்டது", "இன்ஜின் அறையில் தீ", "கப்பலில் தீ எரிகிறது"],
        I.FLOODING: ["படகில் தண்ணீர் புகுகிறது", "கீழே இருந்து தண்ணீர் வருகிறது", "படகு மூழ்குகிறது, தண்ணீர் நிரம்புகிறது"],
        I.ENGINE_FAILURE: ["இன்ஜின் நின்றுவிட்டது", "இன்ஜின் பழுதாகிவிட்டது, நாங்கள் மிதந்து செல்கிறோம்", "இன்ஜின் வேலை செய்யவில்லை"],
        I.CAPSIZING: ["படகு கவிழ்கிறது", "படகு கவிழ்ந்துவிட்டது", "படகு ஒரு பக்கம் சாய்கிறது"],
        I.MAN_OVERBOARD: ["ஒருவர் கடலில் விழுந்துவிட்டார்", "ஒரு ஆள் தண்ணீரில் விழுந்தார்", "எங்கள் நண்பர் ஒருவர் கடலில் விழுந்தார்"],
        I.MEDICAL: ["ஒருவருக்கு உடல்நிலை மோசமாக உள்ளது", "ஒருவருக்கு பலத்த காயம்", "ஒருவர் மயக்கமாக இருக்கிறார்"],
        I.COLLISION: ["வேறு கப்பலுடன் மோதிவிட்டோம்", "ஒரு கப்பல் எங்கள் படகில் மோதியது", "மோதல் ஏற்பட்டது"],
        I.GROUNDING: ["படகு பாறையில் சிக்கிக்கொண்டது", "படகு மணலில் தரை தட்டியது", "படகு அடியில் மோதி சிக்கியது"],
    },
    Language.ML: {
        "opening": ["{CALL} {CALL} {CALL}, ഇത് {V}, {V}", "{CALL}, ഞങ്ങൾ {V} ബോട്ടിൽ നിന്നാണ് വിളിക്കുന്നത്", "{CALL} {CALL}, {V} ആണ് സംസാരിക്കുന്നത്"],
        "opening_noname": ["{CALL} {CALL}, ആരെങ്കിലും കേൾക്കുന്നുണ്ടോ", "{CALL}, കോസ്റ്റ് ഗാർഡ് മറുപടി തരൂ", "{CALL} {CALL}, സഹായം വേണം"],
        "mmsi": ["MMSI {M}", "ഞങ്ങളുടെ MMSI {M}", "MMSI നമ്പർ {M}"],
        "vessel_type": ["ഞങ്ങൾ ഒരു {T} ആണ്", "ഇത് ഒരു {T} ആണ്", "{T}"],
        "position": ["ഞങ്ങളുടെ പൊസിഷൻ {P}", "പൊസിഷൻ {P}", "ഞങ്ങൾ ഉള്ള സ്ഥലം {P}"],
        "relative": ["{L}-ൽ നിന്ന് {D} മൈൽ {C}", "{L}-ൽ നിന്ന് ഏകദേശം {D} മൈൽ {C}", "{L}-ൽ നിന്ന് {D} നോട്ടിക്കൽ മൈൽ {C}"],
        "pob": ["ബോട്ടിൽ {N} പേരുണ്ട്", "ഞങ്ങൾ {N} പേരാണ്", "{N} ആളുകൾ ഉണ്ട്"],
        "assist": ["ഉടനെ സഹായം വേണം", "വേഗം സഹായം അയക്കൂ", "ഞങ്ങൾക്ക് അടിയന്തര സഹായം വേണം"],
        "close": ["ഓവർ", "over", ""],
        "say_again": ["കേൾക്കുന്നില്ല, വീണ്ടും പറയൂ", "എന്ത്? ഒന്നുകൂടി പറയൂ", "ശബ്ദം കേൾക്കുന്നില്ല"],
        I.FIRE: ["ബോട്ടിൽ തീ പിടിച്ചു", "എഞ്ചിൻ റൂമിൽ തീ", "കപ്പലിൽ തീ പടരുന്നു"],
        I.FLOODING: ["ബോട്ടിൽ വെള്ളം കയറുന്നു", "അടിയിൽ നിന്ന് വെള്ളം വരുന്നു", "ബോട്ട് മുങ്ങുന്നു, വെള്ളം നിറയുന്നു"],
        I.ENGINE_FAILURE: ["എഞ്ചിൻ നിന്നുപോയി", "എഞ്ചിൻ കേടായി, ഞങ്ങൾ ഒഴുകി പോകുന്നു", "എഞ്ചിൻ പ്രവർത്തിക്കുന്നില്ല"],
        I.CAPSIZING: ["ബോട്ട് മറിയുന്നു", "ബോട്ട് മറിഞ്ഞു", "ബോട്ട് ഒരു വശത്തേക്ക് ചരിയുന്നു"],
        I.MAN_OVERBOARD: ["ഒരാൾ കടലിൽ വീണു", "ഒരാൾ വെള്ളത്തിൽ വീണു", "ഞങ്ങളുടെ ഒരു കൂട്ടുകാരൻ കടലിൽ വീണു"],
        I.MEDICAL: ["ഒരാൾക്ക് ഗുരുതരമായ അസുഖം", "ഒരാൾക്ക് സാരമായ പരിക്ക്", "ഒരാൾ ബോധം കെട്ടു കിടക്കുന്നു"],
        I.COLLISION: ["മറ്റൊരു കപ്പലുമായി കൂട്ടിയിടിച്ചു", "ഒരു കപ്പൽ ഞങ്ങളുടെ ബോട്ടിൽ ഇടിച്ചു", "കൂട്ടിയിടി ഉണ്ടായി"],
        I.GROUNDING: ["ബോട്ട് പാറയിൽ കുടുങ്ങി", "ബോട്ട് മണലിൽ ഉറച്ചു", "ബോട്ട് അടിയിൽ തട്ടി കുടുങ്ങി"],
    },
}

VESSEL_TYPE_WORDS: dict[Language, dict[VesselType, str]] = {
    Language.EN: {
        VesselType.FISHING_BOAT: "fishing boat", VesselType.TRAWLER: "trawler", VesselType.CARGO: "cargo ship",
        VesselType.TANKER: "tanker", VesselType.FERRY: "passenger ferry", VesselType.YACHT: "yacht", VesselType.TUG: "tug",
    },
    Language.HI: {
        VesselType.FISHING_BOAT: "मछली पकड़ने वाली नाव", VesselType.TRAWLER: "ट्रॉलर", VesselType.CARGO: "मालवाहक जहाज़",
        VesselType.TANKER: "टैंकर", VesselType.FERRY: "यात्री फ़ेरी", VesselType.YACHT: "याट", VesselType.TUG: "टग",
    },
    Language.TA: {
        VesselType.FISHING_BOAT: "மீன்பிடி படகு", VesselType.TRAWLER: "டிராலர்", VesselType.CARGO: "சரக்குக் கப்பல்",
        VesselType.TANKER: "டேங்கர்", VesselType.FERRY: "பயணிகள் படகு", VesselType.YACHT: "சொகுசுப் படகு", VesselType.TUG: "இழுவைப் படகு",
    },
    Language.ML: {
        VesselType.FISHING_BOAT: "മീൻപിടിത്ത ബോട്ട്", VesselType.TRAWLER: "ട്രോളർ", VesselType.CARGO: "ചരക്കുകപ്പൽ",
        VesselType.TANKER: "ടാങ്കർ", VesselType.FERRY: "യാത്രാബോട്ട്", VesselType.YACHT: "യാട്ട്", VesselType.TUG: "ടഗ്",
    },
}

FILLERS: dict[Language, list[str]] = {
    Language.EN: ["uh", "please", "quickly", "help"],
    Language.HI: ["अरे", "जल्दी", "प्लीज़", "भगवान"],
    Language.TA: ["ஐயோ", "சீக்கிரம்", "ப்ளீஸ்", "கடவுளே"],
    Language.ML: ["അയ്യോ", "വേഗം", "പ്ലീസ്", "ദൈവമേ"],
}

# The coast guard side is always English, as on Indian VHF working channels.
MRCC_ASK: dict[str, list[str]] = {
    "vessel_name": ["say name of your vessel", "what is your vessel name"],
    "position": ["say your position", "what is your position"],
    "incident_types": ["say nature of distress", "what is the nature of your distress"],
    "pob": ["how many persons on board", "say number of persons on board"],
}
