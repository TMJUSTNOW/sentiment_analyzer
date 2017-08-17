import pickle
import time
import re

from line_profiler import LineProfiler
from keras.models import model_from_json
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
from nltk.corpus import wordnet
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.keyedvectors import KeyedVectors

##### Analyze these sntence
#indian bank write percent value bad loan large stress asset account hit billion rating agency
# indian benchmark gain most over week lead company
# lg electronics inc thursday continued loss mobile unit limited growth profit percent firm prepare release
# stock jump nearly percent optical maker report rise percent rs period end june drive strong


## Stop Words
stop_words = ['after', 'others', 'clud', 'would', 'id', 'self', 'youll', 'brother', 'second', 'shes','yet','bit', 'break','shop','suppose','sell','holiday',
              'becomes', 'youd', 'etc', 'college', 'shin', 'server', 'uni', 'jordanknight', 'wishing', 'swine', 'adam','full', 'clean','anyway','sometimes',
              'prom', 'england', 'bath', 'battery', 'button', 'plz', 'freakin', 'click', 'addict', 'topic', 'code','puppy', 'app', 'freak', 'random', 'ball', 'heard', 'mile', 'plane', 'revision', 'sat', 'ahead', 'kiss', 'arm', 'ear', 'indeed', 'tan', 'state', 'hmmm', 'group', 'longer', 'vip', 'mommy', 'promise', 'completely', 'fire', 'co', 'sale', 'hotel', 'wash', 'web', 'lakers', 'dark', 'currently', 'ruin', 'lately', 'ohh', 'ff', 'hun', 'left', 'deserve', 'stuck', 'huh', 'track', 'lost', 'buddy', 'profile', 'quick', 'bloody', 'fucking', 'surprise',
              'tough', 'island', 'wave', 'official', 'oops', 'setting', 'pls', 'ok', 'thunder', 'oo', 'grandma', 'fave','complete', 'cross', 'eh', 'bug', 'gosh', 'extra', 'practice', 'travel', 'street', 'design', 'watchin',
              'size', 'oooh', 'charge', 'disney', 'guys', 'xbox', 'mind', 'link', 'fall', 'course','hair', 'wife', 'eye', 'thats', 'come', 'plot',
              'english', 'so', 'house', 'year','stick','nap', 'sigh', 'ride', 'dress', 'sing', 'tea', 'fast', 'account', 'realize', 'park', 'decide', 'understand', 'order', 'nd', 'cream', 'less', 'chat','way','thank','spend','fly', 'try', 'baby', 'finish','already', 'call','eat', 'follow', 'play', 'men', 'take', 'cooky', 'gunna', 'wall', 'tree', 'client',
              'essay', 'sushi', 'germany', 'twice','list', 'be', 'jon', 'loud', 'fav', 'danny', 'boat', 'style','interesting', 'mention', 'camp', 'except', 'chill', 'block', 'near', 'hospital', 'lets', 'radio', 'fight', 'fact', 'lonely', 'lie', 'gorgeous', 'looks', 'alright', 'taste', 'issue', 'storm', 'scary', 'wine',
              'girlfriend', 'vid', 'spring', 'jam','hit','reply', 'close', 'remember', 'soup', 'line', 'put','lines', 'direct', 'three', 'couple', 'next','pray', 'glass', 'waiting', 'count', 'arrive', 'bag', 'behind', 'bar', 'invite', 'mothers', 'pull', 'sounds', 'reading', 'remind', 'along', 'round', 'matter', 'although', 'swim', 'guitar', 'mate', 'joke', 'return', 'low', 'single', 'hangover', 'everybody', 'door', 'hahah', 'bb',
              'home', 'else', 'saw', 'work','check', 'foot','may', 'birthday','bottle', 'machine', 'sunburn', 'mmm', 'grad', 'mmm', 'available','traffic',
              'followers', 'chris', 'joy','anymore', 'teach','got','catch', 'hang','outside', 'drive', 'real', 'sit', 'feeling', 'every', 'turn', 'follower','going', 'send', 'listen','stay', 'god', 'kitty', 'kate', 'mcfly', 'demi', 'france', 'davidarchie', 'tip',
              'gutted', 'ima', 'present', 'airport','computer','half','hello', 'able','room','chance', 'throat','pool','band','probably', 'bring', 'anyone', 'babe','lunch', 'food','cry','whats', 'seem', 'update', 'meet', 'forget', 'uh', 'attack', 'woohoo', 'mama', 'local', 'jack',
              'student', 'smoke', 'brazil', 'york','everything', 'taking', 'couch','green', 'mitchelmusso', 'support', 'file','flight', 'comment', 'kick', 'sunshine', 'beat', 'butt', 'alive', 'imma','bff', 'texas', 'morrow', 'pant', 'law', 'cloud', 'desk',
              'personal', 'kitchen', 'hes', 'west', 'push', 'king', 'training','warm', 'huge', 'bet','paper', 'card', 'burn', 'hold', 'load', 'cook', 'fell', 'rather', 'evening', 'moon', 'apparently','ate', 'boys', 'wii', 'strange','credit', 'mi', 'girls', 'strawberry', 'control', 'grade', 'talent', 'anytime', 'forgot','burnt', 'brothers', 'connect', 'accident',
              'taylorswift', 'switch', 'mo', 'made', 'canada', 'jb', 'mornin', 'fry','bus', 'cake','blue', 'white', 'age', 'throw', 'plus', 'nope', 'small', 'looking', 'forever','trying','vacation', 'beer', 'upload', 'listening', 'mess', 'working', 'light', 'day', 'channel', 'bird','cleaning', 'cell', 'wheres', 'schedule', 'jump', 'msn', 'folk', 'article', 'milk', 'imagine', 'key', 'darn', 'dnt', 'san', 'nervous', 'making', 'knee', 'clear', 'everyday', 'crack', 'anywhere', 'afford', 'consider', 'floor', 'however', 'somebody', 'expensive', 'secret', 'drag', 'draw', 'quote', 'lake', 'chip', 'unless', 'strong', 'delicious', 'row', 'system', 'crappy', 'workout', 'create',
              'husband', 'blackberry', 'gig', 'goes', 'fam', 'guy', 'allow', 'bro', 'alot', 'whatever', 'roll', 'clothes','theres', 'minutes', 'done','july', 'different', 'safe', 'record', 'lay', 'sort', 'cancel', 'relax', 'thinking', 'stand', 'absolutely','apart', 'cuddle', 'hide', 'mini', 'ily', 'explain', 'j', 'attempt', 'possibly', 'bos', 'nooo', 'war', 'pet', 'sam', 'flower', 'difficult', 'lead', 'option', 'juice', 'spanish', 'table', 'rough', 'handle', 'lookin', 'fault', 'dat', 'public', 'nightmare', 'football', 'hehehe', 'soccer', 'request', 'everyones', 'florida', 'squarespace', 'sport', 'six', 'onto', 'sa', 'hoo', 'heres', 'north', 'winter', 'alex', 'delay', 'truly', 'medium', 'spam', 'smart', 'workin', 'wind', 'pro',
              'world', 'could', 'yourselves', 'text', 'august', 'energy', 'ad', 'egg', 'tonite', 'narrative', 'point','appreciate', 'speak', 'cover', 'grow','major', 'steal', 'shout', 'trailer', 'thru', 'peep', 'wet', 'celebrate', 'offer', 'copy', 'trek', 'exhaust', 'friends', 'depress', 'straight', 'confuse', 'shift', 'badly', 'contact', 'woke', 'honey', 'screw', 'drinking', 'epic', 'spot', 'tune', 'graduate', 'gift', 'jk', 'australia', 'price', 'board', 'south', 'hahahaha', 'fresh', 'match', 'prob', 'bunch', 'cough', 'laundry', 'color', 'afraid', 'degree', 'include', 'feelin', 'anyways', 'sum', 'broken','mix', 'living', 'prepare', 'dry','orange', 'attend', 'boston', 'daily',
              'everyone', 'sequence', 'deliver', 'finale', 'sit', 'assignment', 'spell', 'science', 'social', 'treat','heading', 'fml', 'nearly', 'fat', 'somewhere', 'playing', 'nose', 'child', 'f', 'tweetdeck', 'earlier','ton', 'receive', 'library', 'lots','gah', 'hah','whos', 'grrr', 'lord', 'suggestion','sandwich', 'ac', 'bread', 'mostly', 'aunt', 'shitty', 'conference', 'deep', 'tha', 'awhile', 'spent', 'relaxing', 'ms', 'model', 'noooo', 'cupcake', 'net', 'fab', 'network', 'fuckin', 'prefer', 'toast', 'lmfao', 'sarah', 'status', 'otherwise', 'aha', 'os', 'heh', 'interested', 'abt', 'appear', 'sucks', 'ughh', 'tattoo', 'rent', 'earth', 'meal', 'hoping', 'bowl', 'golf', 'cali', 'allergy', 'attention', 'macbook', 'anybody', 'gross', 'heck', 'snow', 'contest', 'slightly', 'cable',
              'us', 'weight', 'level', 'exams', 'million', 'america', 'diet', 'getting', 'toe', 'stage', 'dammit','rd', 'shut', 'fever', 'delete', 'better', 'race', 'silly', 'info', 'blow', 'ring', 'went', 'magic', 'brain', 'lesson', 'none', 'manage', 'background', 'possible', 'doubt', 'missing', 'land', 'interest', 'sitting', 'nail', 'bite', 'mark', 'doin', 'normal', 'starbucks', 'rid', 'middle','cheap',
              'yum', 'taylor', 'gay', 'sooooo', 'daddy', 'eating', 'dang', 'awwww', 'nyc', 'paint', 'album', 'ny','sometime','neck','player','simple', 'cost', 'prayer', 'upgrade', 'barely', 'everywhere','program', 'finished', 'somehow', 'hav', 'matt', 'chinese', 'edit', 'found', 'angel', 'harry', 'marry', 'log', 'ahhhh', 'atm',
              'jon', 'series', 'store', 'actually', 'years', 'man', 'im', 'with', 'only', 'is', 'had','rd', 'shut', 'fever', 'delete', 'better', 'race', 'silly', 'info', 'blow', 'ring', 'went', 'magic', 'brain', 'lesson', 'none', 'manage', 'background', 'possible', 'doubt', 'missing', 'land', 'interest', 'sitting', 'nail', 'bite', 'mark', 'doin', 'normal', 'starbucks', 'rid', 'middle',
              'll', 'into', 'being', 'a', 'ohhh', 'me', 'seem', 'has', 'mine', 'ask', 'age', 'dinner', 'been', 'why','force', 'flat', 'connection', 'apartment', 'market', 'limit', 'leaving', 'perhaps', 'loss', 'slept', 'rule','goodness', 'total', 'um', 'paris', 'bgt', 'h', 'festival', 'shake', 'woot', 'dc', 'skin',
              'where', 'can', 'than', 'in', 'sex','memory', 'blood', 'pc', 'nick', 'itunes', 'coz', 'cheese', 'chillin', 'hill', 'mall', 'dunno', 'mail', 'bear', 'nobody', 'def', 'ooh', 'drunk', 'waste', 'choice', 'heat', 'sky','surgery', 'sir', 'quit', 'across', 'hero', 'pair',
              'days', 'from', 'who', 'the', 'should', 'your', 'blah', 'everything', 'thanx', 'weve', 'xo', 'hannah',
              'jonathanrknight', 'bf', 'mobile', 'tweeps', 'vegas', 'teacher', 'realise', 'kno', 'shopping', 'area','purse', 'victoria', 'alyssamilano',
              'outta', 'bother', 'dentist', 'session', 'pack', 'themselves', 'chicago', 'donniewahlberg', 'while', 'we','africa', 'sm', 'injury', 'necklace', 'remote', 'lens', 'indoors', 'amen', 'sd', 'device', 'bound', 'sofa',
              'against', 'further','sydney','conan','sleeping', 'dave','soul','grab', 'recently', 'tennis', 'software', 'makes', 'wide', 'remove', 'enter', 'replace', 'awe'
              'these', 'those', 'were', 'be', 'holy', 'turn', 'leave', 'station', 'midnight', 'him', 'cuz','hardcore', 'sorta', 'extreme', 'arse', 'mario', 'relieve', 'coverage', 'phoenix', 'linux', 'purpose', 'therapy', 'sonic', 'legal', 'safely', 'dizzy', 'simon', 'introduce', 'karma', 'halo', 'rabbit', 'cheesecake', 'buying', 'snl', 'phil', 'distance', 'signing', 'nigga',
              'teeth', 'bummer', 'office', 'argh', 'our', 'which', 'yo', 'jus', 'piss', 'daughter', 'cup', 'wit','effin', 'eve', 'yogurt', 'plastic', 'sand', 'needle', 'mariahcarey', 'valley', 'tiger', 'wings', 'clever', 'beg', 'legend', 'united', 'nine', 'edition', 'gt', 'sober', 'rly', 'twitterville', 'workshop', 'usb', 'mirror', 'z', 'tad', 'election', 'aaron', 'bagel', 'proof', 'came', 'met', 'dia', 'ian', 'addictive', 'smooth', 'chain', 'backyard', 'await', 'countdown', 'development', 'sweden', 'cleveland', 'trentreznor', 'inbox', 'chilly', 'reception', 'jessica', 'patrick', 'overnight', 'engine', 'tournament', 'beef', 'tat', 'tweetup', 'federer', 'hahaa', 'doh', 'thread', 'thou',
              'just', 'box', 'note', 'tom', 'myspace', 'service', 'cheer', 'raining', 'road', 'rainy', 'window','menu', 'butterfly', 'sickness', 'dvr', 'fantasy', 'started', 'chop', 'besties', 'alice', 'management', 'berry', 'mega', 'srsly', 'skirt', 'yelyahwilliams', 'oooo', 'vanilla', 'potential', 'xxxxx', 'factory', 'korean', 'memorial', 'sara', 'kat', 'neglect', 'whale', 'meeee', 'ikea', 'passion', 'don', 'trap', 'iz', 'polish', 'active', 'leavin', 'reckon', 'pat', 'videos', 'pr', 'gum', 'knw', 'ubuntu', 'blogger', 'shaun', 'girly', 'strip', 'diff', 'twist', 'collect', 'convo', 'detroit', 'campus', 'laker', 'benefit', 'hype', 'iced', 'nokia', 'furniture', 'uber', 'humid', 'exit', 'opposite', 'solid', 'chilling', 'yucky', 'childhood',
              'tummy', 'doing', 'this', 'had','mike', 'shoulder', 'salad', 'asap', 'dirty', 'performance', 'comp', 'shell', 'driver', 'grand', 'error', 'member', 'freeze', 'spending', 'obviously', 'goodmorning', 'hook', 'survive', 'user', 'literally', 'twitpic', 'scratch', 'chick', 'crush', 'knock', 'diversity', 'toy', 'massive', 'everytime', 'mmmm', 'wee', 'letter', 'driving', 'apps', 'bake', 'boot', 'plant', 'pa', 'writing', 'stream', 'trend', 'uncle', 'presentation', 'conversation', 'susan', 'wwwtweeteraddercom', 'sugar', 'hella', 'brand', 'baseball', 'crave', 'tweeter', 'montana', 'horse', 'third', 'vet', 'yard', 'ankle', 'tht', 'truck', 'accept', 'theyll', 'aim', 'original', 'quality', 'fellow', 'andy',
              'theirs', 'i', 'under', 'm', 'herself', 'show', 'club', 'alone', 'bitch', 'cheap', 'interview', 'yummy','spread', 'hole', 'doc', 'california', 'jesus', 'launch', 'lovin', 'lip', 'print', 'ryan', 'wifi', 'dougiemcfly', 'detail', 'selenagomez', 'soooooo', 'perform',
              'type', 'bbq', 'apple', 'graduation', 'by', 'at', 'until', 'here', 'officially', 'case', 'idk', 'bike','atlanta', 'ang', 'bone', 'mon', 'ng', 'careful', 'vs', 'bbc', 'surf', 'jacket', 'compliment', 'yeahh', 'glorious', 'iv', 'brownie', 'speech', 'allergic', 'twitterverse', 'bedroom', 'lightning', 'bruise', 'bob', 'similar', 'nooooo', 'normally', 'marketing', 'sunburnt', 'needs', 'pretend', 'trade', 'roast', 'offline', 'emily', 'en', 'talented', 'bull', 'playin', 'highly','hack', 'package', 'basketball', 'competition', 'bt', 'simply', 'hay', 'rich', 'mexico', 'bubble', 'gd', 'singer', 'punch', 'rofl', 'cereal', 'att', 'yey', 'battle', 'gaga', 'awh', 'virus', 'pump', 'grill', 'ireland', 'pure', 'soak', 'gf', 'que', 'prolly',
              'itll', 'garden', 'sims', 'leg', 'boyfriend', 'camera', 'jonasbrothers', 'son', 'were', 'if', 'myself','closer', 'hawaii', 'pee', 'itd', 'poker', 'court', 'sauce', 'icecream', 'ace', 'surely', 'perezhilton', 'apt', 'lolz', 'china', 'easily', 'bought', 'bugger', 'distract', 'boooo', 'ure', 'pride', 'desperate', 'mexican', 'assume', 'fridge', 'seeing', 'stalk', 'cherry', 'wondering', 'awwwww', 'preview', 'common', 'mass', 'oil', 'ti', 'feedback', 'effort', 'india', 'female','skill', 'september', 'babysitting','outfit', 'trick', 'hd', 'wolverine', 'fox', 'gain', 'eu', 'piano', 'convince', 'frm', 'urgh', 'pity', 'bestie', 'swimming', 'funeral', 'rate', 'jordan', 'stone', 'fully', 'studying', 'spoil', 'monster', 'chest', 'tony', 'yoga', 'youngq', 'con', 'thnx', 'faster', 'confused', 'deck', 'momma', 'cap', 'creepy', 'opinion', 'stoke', 'garage',
              'ma', 'delivery','password', 'swell', 'pressure', 'il', 'progress', 'mp', 'earn', 'pill', 'spin', 'bud', 'hrs', 'pasta', 'entry', 'steak', 'comfy', 'technology', 'pillow', 'hockey', 'di', 'escape', 'wheel', 'donut', 'damnit', 'dare', 'philippines', 'repair', 'softball', 'ch', 'accidentally', 'tasty', 'scotland', 'audition', 'register','melbourne', 'wana', 'recipe', 'freedom', 'dis', 'skool','marathon', 'productive','dear','join','together','figure','sunny', 'learn','test','watching','tell',
              'ours', 'as', 'all', 'each', 'how', 'when', 'adult', 'thanks', 'whose', 'other', 'itself', 'an','beginning', 'cramp', 'plug', 'sittin', 'angels', 'mary', 'sneak', 'faith', 'houston', 'pig', 'yu', 'jimmy', 'friendly', 'tomorow', 'farrah', 'compare', 'iranelection', 'doll', 'november', 'moms', 'tweets', 'sweat', 'austin', 'icon', 'maths', 'yikes', 'metro', 'ghost', 'bk', 'manchester', 'announce', 'april', 'makin', 'square', 'walking', 'tt','access', 'coast','advice', 'lock','happen',
              'thursday', 'ahhh', 'miley', 'my', 'did', 'them', 'ourselves','va', 'melt', 'dawn', 'tomfelton', 'cooking', 'gloomy', 'eric', 'fest', 'apprentice', 'refresh', 'manager', 'kim', 'philly', 'opportunity', 'struggle', 'cam', 'starve', 'yawn', 'tax', 'tongue', 'bump', 'meat', 'contract','cop', 'shaundiviney', 'gfalcone', 'item', 'sadness', 'yuck', 'poster', 'chelsea', 'noon', 'heel', 'pot', 'moro', 'desktop', 'depend', 'confirm', 'lawn', 'omfg', 'q', 'kris', 'sock', 'ray', 'serve', 'peanut', 'hungover', 'aswell', 'japan', 'singing', 'ass', 'walmart', 'noodle', 'naked', 'cnt', 'lover', 'brunch', 'italian', 'potato', 'couldve', 'community', 'honor', 'pen', 'hip', 'mia', 'ho', 'gs',
              'during', 'whom', 'am', 'o', 'same', 'are', 'visual', 'maintain', 'number', 'wannabe', 'jonas', 'bc','lab', 'whilst', 'nkotb', 'mondays', 'noo', 'theyve', 'thinkin', 'hills', 'scared', 'muscle', 'snack', 'dan', 'poo', 'lyric', 'impossible', 'purchase', 'relate', 'application', 'shape', 'monkey', 'rehearsal', 'van', 'clip', 'vids', 'katie', 'mee', 'charger', 'section', 'celebrity', 'ol', 'yellow', 'justin', 'giant', 'insomnia', 'tim', 'general', 'mission', 'kelly', 'jess', 'yell', 'bio', 'comic', 'breath', 'laying', 'planning', 'mcdonalds', 'grass', 'comin', 'coming',
              'french', 'business', 'yup', 'tuesday', 'have', 'ipod', 'above', 'what', 'both', 'do', 'off', 'before','spirit', 'noone', 'blink', 'tool', 'spider', 'someday', 'ocean', 'darling', 'actual','ow', 'duty','picnic','apply', 'loose', 'candy', 'er', 'proper', 'idiot', 'migraine', 'situation', 'shock', 'cancer', 'breathe', 'meh', 'certainly', 'ashley', 'mm', 'tooth', 'med', 'texting', 'rite', 'cash', 'burger', 'challenge', 'cavs', 'whenever', 'flash', 'heaven', 'bacon', 'killer', 'worried', 'xxxx', 'chuck', 'todays', 'packing', 'large', 'kitten',
              'or', 'having', 'now', 'too', 'on', 'and', 'town', 'ms', 'cd', 'ode', 'imagery', 'company', 'xoxo','cyrus', 'amy', 'wrap','shouldve', 'ie', 'al', 'depressing', 'gb', 'nothin', 'tie','tryin', 'makeup', 'hunt', 'museum', 'jason', 'field', 'tag', 'corner', 'bathroom', 'dish', 'rice', 'fancy', 'lauren', 'potter', 'missin', 'booo', 'truth', 'sea', 'process', 'commercial', 'tech', 'rubbish', 'hadnt', 'banana', 'yah', 'difference', 'response', 'repeat', 'bedtime', 'chair', 'loser', 'national', 'fee', 'appointment', 'pancake',
              'doctor', 'wedding', 'dm', 'wednesday', 'twilight', 'luv', 'notice', 'ps', 'smell', 'front', 'hubby','gb', 'nothin', 'tie', 'recommendation','george', 'dread', 'tommorow', 'cookie', 'stink', 'cheers', 'gold', 'vega', 'joeymcintyre', 'mouse','anniversary', 'ground', 'press', 'ftw', 'certain', 'fathers', 'retweet', 'sux', 'gas', 'atl', 'neighbor', 'kevin', 'rush', 'el','httptweetsg', 'health', 'havin', 'ahaha', 'tix','nah', 'often', 'revise', 'twit', 'fish', 'fill', 'double', 'feed', 'bank', 'wat',
              'shot', 'through', 'ppl', 'mood', 'meeting', 'lil', 'she', 'his', 'do', 'bday', 'laptop', 'youtube','wen','peterfacinelli', 'usa', 'guilty', 'miami', 'mountain', 'tried', 'pub', 'youu', 'miserable', 'orlando', 'massage', 'ed', 'pre', 'pie', 'current', 'settle', 'tight', 'address', 'keyboard', 'bang', 'install', 'princess', 'butter', 'ebay', 'adventure', 'wicked', 'paul', 'wants', 'times', 'seattle', 'clock', 'nephew', 'forum', 'twin', 'running', 'dallas', 'instal', 'colour', 'hanging', 'niece', 'tmrw', 'wing', 'aka', 'nut', 'iran', 'pour', 'clue', 'purple', 'coke', 'disappear', 'ep', 'mode', 'involve', 'eff', 'ew', 'parade', 'starting', 'magazine', 'atleast', 'crew', 'chapter', 'fruit', 'italy', 'ex', 'rub', 'jersey', 'dj', 'tweeting', 'building', 'terminator', 'wwwtweeterfollowcom', 'le', 'ughhh', 'heyy', 'bell', 'suggest', 'alcohol', 'thunderstorm', 'curious', 'pound', 'regular', 'correct', 'bust', 'language', 'josh', 'basically', 'tshirt',
              'church', 'side', 'mother', 'homework', 'goin', 'tour', 'lady', 'there', 'won', 'they','firefox', 'probs', 'data', 'sf', 'peter', 'snap', 'root', 'watched', 'shine', 'insane', 'flip', 'talkin', 'swift', 'seems', 'hugs', 'bah',
              'to', 'are', 't', 'few', 'put', 'was', 'woo', 'within', 'gang', 'wellacted', 'stir', 'bring', 'soap','aplusk', 'climb', 'plenty', 'wooo', 'wouldve', 'tooo', 'private', 'content', 'bee', 'guest', 'dig',
              'gym', 'shirt', 'wed', 'ache', 'si', 'chicken', 'goodbye', 'yr', 'g', 'pizza', 'talk', 'today', 'hows','broke', 'tweetie', 'cruise', 'ko', 'whether', 'ooo', 'boyle', 'hayfever', 'beyond', 'quiz', 'cafe',
              'c', 'mum', 'episode', 'sleepy', 'hr', 'via', 'xxx', 'london', 'ahh', 'chocolate', 'phone', 'date', 'red','gossip', 'yeh', 'besides', 'dannymcfly', 'hw', 'bunny', 'beta','tiny', 'thankyou', 'txt', 'jay', 'pas', 'smh', 'restaurant', 'senior', 'grey', 'east', 'feels', 'hardly', 'focus', 'research', 'ish', 'dollar', 'admit', 'center', 'discover', 'cos', 'grr', 'goal', 'gr', 'ben', 'windows', 'decision', 'german', 'properly', 'ship', 'grocery', 'speed', 'host', 'jean', 'europe', 'cloudy', 'slowly', 'customer', 'downtown', 'decent', 'ty', 'haircut', 'taco',
              'city', 'soooo', 'nite', 'xd', 'ice', 'bout', 'it', 'did', 'june', 'cat', 'afternoon', 'tommcfly', 'hop','gnight','britney', 'bay', 'inspire', 'suit', 'queen', 'avatar', 'exercise',
              'lmao', 'himself', 'facebook', 'aw', 'fix', 'asleep', 'her', 'such', 'have', 'yours', 'more', 'for','things', 'nowhere', 'gear', 'infection', 'msg', 'palm', 'noise', 'several', 'geek', 'wild', 'exist', 'spain', 'iron', 'bomb', 'zoo', 'toronto', 'suddenly', 'umm', 'bean', 'took', 'steve', 'per', 'buck', 'river', 'stock', 'headed', 'whoa', 'skype',
              're', 'will', 'ain', 's', 'you', 'identity', 'leaden', 'christmas', 'americans', 'street', 'preposterous','ladies', 'petewentz', 'lion', 'randomly', 'katy','centre', 'thankfully',
              'l', 'timing', 'pink', 'audience', 'viewer', 'search', 'get', 'toward', 'gettin', 'ouch', 'bum', 'cousin','curl', 'prince', 'gona', 'wood', 'thurs', 'dancing', 'woah', 'singapore', 'sry', 'zone', 'popular', 'curse', 'edward', 'teen', 'sundays', 'eventually', 'url', 'sniff', 'sooner', 'jeff', 'meant', 'bein', 'theory', 'paranoid', 'jet', 'license', 'ups', 'se', 'moving', 'bleh', 'writer', 'cmon', 'amanda', 'rt', 'errand', 'microsoft', 'havnt', 'dye', 'blind', 'luckily', 'lvatt', 'gun', 'tomato', 'ban', 'blonde', 'demo', 'function', 'roof', 'wallet', 'andrew', 'tap', 'shave', 'grandpa', 'muffin', 'layout', 'photoshop', 'split', 'bros', 'hall', 'nighty', 'podcast', 'grace', 'subway', 'thatll', 'hire', 'woop', 'incredibly', 'diego', 'headphone', 'windy', 'spare', 'zombie', 'tryna', 'mah', 'planet', 'rachel', 'panda', 'hooray', 'tease', 'police', 'hee', 'worse', 'destroy', 'transformer', 'university', 'spill', 'rescue', 'drum', 'panic', 'tgif', 'photography', 'yahoo', 'hangin', 'kirstiealley', 'cod', 'bummed', 'knight', 'officialtila', 'amazon', 'sweetheart', 'security', 'lemon', 'sheet','oven', 'pe', 'loop', 'dope', 'dust', 'holla', 'toooo', 'pit', 'eatin', 'poem', 'geography', 'sub', 'tweeple', 'frozen', 'recital', 'pin', 'hike', 'vista', 'helpful', 'vibe', 'kimkardashian', 'dump', 'mental', 'nj', 'pregnant', 'thomas', 'marley', 'dannygokey', 'friggin', 'eminem', 'argentina', 'whoo', 'sucky', 'society', 'bracelet', 'torture', 'hates', 'nc', 'asshole', 'therealjordin', 'whoever', 'upon', 'aussie', 'march', 'voting', 'flow', 'organize', 'pix', 'grateful', 'pleased', 'favor', 'nerve', 'precious', 'te', 'hood', 'advance', 'swollen', 'gots',
              'uk', 'stomach', 'shoe', 'google', 'their', 'yall', 'fb', 'team', 'flu', 'thx', 'wtf', 'voice', 'annoy','esp', 'fri', 'central', 'bing', 'universe', 'jerk', 'tank', 'coach', 'duck','coursework', 'meee', 'seven', 'tube', 'mrs', 'swing', 'takin', 'tape', 'reunion', 'waitin', 'owe', 'sudden', 'tbh', 'itchy', 'judge', 'disneyland', 'youuu','wk', 'digital', 'hv', 'amuse', 'ohhhh', 'neighbour', 'jen', 'appt', 'turkey', 'alexalltimelow', 'alas', 'dick', 'lisa','heal', 'rainbow', 'talking', 'megan', 'following', 'bleed', 'fl', 'remain', 'divorce', 'afterwards', 'finals', 'gnite', 'tracecyrus', 'awkward', 'fo', 'wth', 'playlist', 'michelle', 'mag', 'yer', 'flag', 'information', 'shiny','nxt', 'jake', 'blanket', 'stair', 'ahah', 'deadline', 'fawcett', 'scott', 'robin', 'exact', 'legs', 'twitters', 'sunglass', 'santa', 'smith', 'learning', 'ellen', 'tues', 'veggie', 'unable', 'tall', 'jim', 'married', 'cartoon', 'steph', 'tx', 'heap', 'asot', 'bowling', 'sob', 'plate', 'gal', 'pete', 'map', 'reaction', 'recent', 'wknd', 'willing', 'coldplay', 'papa', 'thai', 'stranger', 'waking', 'reader', 'approve', 'norway', 'develop', 'concern', 'irish', 'kristen', 'slide', 'raw', 'forecast',
              'finger', 'mtv', 'math', 'followfriday', 'yep', 'download', 'bye', 'rain', 'about', 'between', 'wan','flickr','october', 'hahahah', 'previous', 'wanted', 'cheat', 'source', 'cricket', 'transformers', 'comedy', 'borrow', 'theatre', 'cab', 'location', 'require', 'genius', 'unfair', 'dam', 'jst', 'kids', 'rove', 'pocket', 'useful', 'morn','rockin', 'charlie', 'bak', 'owner', 'definately', 'greek', 'insurance', 'ran', 'hater', 'mmmmm', 'pissed', 'youth', 'solo', 'zero', 'ull', 'iamdiddy', 'trending', 'whoop', 'ga', 'addiction', 'crawl', 'lem', 'wrist', 'gmail', 'ne', 'kobe', 'emma', 'homemade', 'carpet', 'role', 'brace', 'hmmmm', 'parking', 'wise', 'dun', 'ooooh', 'perfectly', 'xp', 'porn', 'accent', 'twittering', 'mummy', 'exchange', 'jamie', 'mow', 'wossy', 'buzz', 'shoutout', 'gone', 'disappointing', 'ash', 'aye', 'rat', 'speaker', 'dads', 'budget', 'geez', 'ashleytisdale', 'indian', 'skate', 'flood', 'sync', 'whore', 'demand', 'solve', 'oz', 'hiya', 'twitterland', 'metal', 'sooooooo', 'guide', 'tmr', 'lilyroseallen', 'cutie', 'rap', 'impressed', 'backup', 'shade', 'charm', 'slice', 'sec', 'speaking', 'doggy', 'dull', 'bastard', 'trash', 'resist', 'annoyed', 'wrk', 'natural', 'shortly', 'joey', 'adore', 'slip', 'brush', 'scare', 'unfollow', 'ds', 'premiere', 'promote', 'intense', 'ohio', 'celebration', 'temp', 'bride', 'anna','hp',
              'hug', 'season', 'tweet', 'hey', 'weekend', 'that', 'once', 'does', 'shall', 've','save', 'drop', 'answer','staff', 'grrrr', 'lounge', 'shed','asian', 'wordpress', 'poop', 'vampire', 'prove', 'dd', 'wireless', 'musicmonday', 'waffle', 'sneeze', 'logo', 'comfort', 'opening', 'express', 'gud', 'vodka', 'brian', 'lang', 'obsess', 'medicine', 'tidy', 'soft', 'hon', 'laura', 'browser', 'bridge', 'bella', 'stephenfry', 'cu', 'snuggle', 'billy', 'thankful', 'motivation', 'lap', 'nicole', 'audio', 'nurse', 'drain', 'somethin', 'biology', 'lb', 'emotional', 'sayin', 'ahhhhh', 'bass', 'maintenance', 'towards', 'semester', 'tempt', 'nicely', 'wars', 'ummm', 'selena', 'plain', 'leak', 'yayy', 'din', 'sausage', 'cow', 'fyi', 'nba', 'envy', 'tee', 'typical', 'damage', 'task', 'thingy', 'kyle', 'smash', 'impress', 'exhausted','personally', 'dannywood', 'andyclemmensen', 'wa', 'johncmayer', 'giggle', 'jones', 'supernatural', 'position', 'inch', 'discuss', 'prize', 'thee', 'yayyy', 'nadal', 'theyd', 'pathetic', 'christian', 'denver', 'phew', 'obama', 'lo', 'grandparent', 'male', 'pup', 'blister', 'easter', 'cycle', 'gg', 'weed', 'blogging', 'jazz', 'belong', 'grind', 'farm', 'physic', 'signal', 'un', 'army', 'username', 'highlight', 'display', 'sleepover', 'patient', 'wud', 'accord', 'explode', 'updated', 'downstairs', 'chore', 'lebron', 'solution', 'skinny', 'refer', 'daniel', 'curry', 'twitterer', 'minus', 'reference', 'transfer', 'thousand',
              'he', 'of', 'y', 'own', 'again', 'roll', 'international', 'medium', 'martin', 'make', 'd', 'people','condition', 'demons', 'supply', 'eww', 'paramore', 'whew', 'concentrate', 'reminder', 'km', 'souljaboytellem', 'songzyuuup', 'shoulda', 'glasgow', 'dublin', 'tend', 'johnny', 'naw', 'spymaster', 'claim', 'wreck', 'grumpy', 'statement', 'activity', 'duh', 'pal', 'closet', 'gooood', 'mint', 'teh', 'stack', 'monitor', 'ft', 'pepper', 'craving', 'balance', 'wipe', 'homie', 'leno', 'caffeine', 'jacob', 'blip',
              'amount', 'screen', 'sun', 'due', 'em', 'mac', 'page', 'yea', 'message', 'xox', 'sony', 'mosquito', 'junior', 'sinus', 'ap', 'remix', 'roommate', 'wire', 'died', 'donate', 'gotten', 'pt', 'league', 'warn', 'checking','any', 'partner','game', 'hehe', 'st','thatd', 'habit', 'markhoppus', 'flick', 'kay', 'nevermind', 'multiple', 'holly', 'twitterberry', 'chem', 'somewhat', 'domain', 'oclock', 'virtual', 'titanic', 'wats', 'lily', 'wishes', 'silver', 'earthquake', 'rose', 'route', 'sent', 'dessert', 'puke', 'freezing', 'emo', 'wah', 'hint', 'ttyl', 'emergency', 'everythings', 'mango', 'pile', 'sean', 'hed', 'umbrella', 'safari',
              'train', 'til', 'ticket', 'pic', 'study', 'damn', 'morning', 'tonight', 'thing', 'twitter', 'haha','patch', 'laurenconrad', 'hahahahaha', 'volunteer','string', 'stephen', 'vancouver', 'ant', 'kit', 'prop', 'salt', 'bash', 'listenin', 'sending', 'ako', 'port', 'gm', 'shud', 'cs', 'wt', 'restore', 'castle', 'circle', 'fridays', 'arse'
              'tomorrow', 'does', 'was', 'its', 'below', 'hers', 'yourself']
movie_stop_words = ['actors', 'actor', 'idea', 'ca', 'provide', 'old', 'timeless','goood', 'cultural', 'collection', 'target',
                    'return', 'want', 'example', 'filmmaker', 'entertainment', 'touch', 'feature', 'use', 'care', 'dvd',
                    'omg', 'wow', 'minute', 'heart', 'set', 'experience', 'piece', 'act', 'also', 'script', 'action',
                    'end', 'cast', 'documentary', 'see', 'life', 'exam', 'cold', 'okay', 'parent', 'k', 'e', 'project',
                    'hmm', 'website', 'da', 'hat', 'youve', 'tho', 'weather', 'hi', 'car', 'post', 'friday', 'kid',
                    'sunday', 'party', 'read', 'story', 'x', 'ur', 'lol', 'yay', 'night', 'na', 'gon', 'villain',
                    'acting', 'u', 'family', 'oh', 'cinematic',
                    'dialogue', 'hm', 'chynna', 'screenplay', 'head', 'popcorn', 'goofy', 'drive', 'simpleminded',
                    'seat', 'part', 'event', 'nature', 'sustain', 'routine', 'sentimental', 'step', 'instantly',
                    'political', 'raise', 'thoroughly', 'engage', 'summer', 'sequel', 'bogarde', 'moment', 'history',
                    'ya', 'thoughtful', 'become', 'photo', 'fan', 'keep', 'ah', 'place', 'write', 'filmmaking',
                    'subject', 'art', 'worth', 'young', 'picture', 'ddlovato', 'p', 'fuck', 'shit', 'plan', 'iphone',
                    'video', 'month', 'aww', 'ta', 'n', 'yesterday', 'mom', 'ugh', 'shampoo', 'painfulbr', 'coolio',
                    'musclebound',
                    'baloney', 'hairline', 'joe', 'blog', 'belly', 'aweinspiring', 'power', 'capture', 'herzog',
                    'lawrence', 'et', 'opera', 'product', 'score', 'email', 'mileycyrus', 'breakfast', 'dude', 'effect',
                    'shower', 'dance', 'visit', 'wear', 'site', 'pick', 'ha', 'concert', 'online', 'sexy', 'soo',
                    'sister', 'internet', 'boo', 'btw', 'news', 'japanese', 'r', 'name', 'drink', 'b', 'hahaha',
                    'monday', 'think', 'cinematography', 'father', 'say',
                    'studio', 'boy', 'direction', 'remind', 'toilet', 'painting', 'cinema', 'evening', 'soundtrack',
                    'design', 'dreadful', 'pm', 'class', 'mouth', 'stoop', 'occasionally', 'animal', 'friend', 'word',
                    'awww', 'imax', 'conviction', 'goodnight', 'trip', 'writ', 'movies', 'american', 'actress',
                    'classic', 'star', 'directed', 'spiritual', 'crime', 'wry', 'final', 'inane', 'directorial',
                    'build', 'fashion', 'provocative', 'add', 'moviemaking', 'junk', 'gem', 'stretch', 'scream',
                    'harvard', 'crowd', 'simultaneously', 'refreshingly', 'future', 'sight', 'doze', 'open', 'result',
                    'heavyhanded', 'offbeat', 'touching', 'visuals', 'dream', 'material', 'bor', 'emerge', 'title',
                    'coffee', 'sooo', 'kinda', 'beach', 'dog', 'saturday', 'xx', 'dad', 'human', 'enterta', 'ett',
                    'production',
                    'performances', 'wish', 'rise', 'feces', 'base', 'examination', 'felt', 'deal', 'flimsy',
                    'demonstrate', 'kill', 'seek', 'air', 'share', 'totally', 'execute', 'films', 'face', 'firstrate',
                    'allen', 'skip', 'element', 'lifeless', 'detailed', 'franchise', 'combine', 'quiet', 'inside',
                    'treasure', 'pop', 'landscape', 'usually', 'superficial', 'standard', 'dramatically', 'characters',
                    'writerdirector', 'oldfashioned', 'help', 'character', 'oscar', 'adaptation', 'tv', 'main', 'sense',
                    'woman', 'girl', 'middle',
                    'scenes', 'terest', 'substance', 'reach', 'newcomer', 'graphic', 'country', 'moviegoing', 'vision',
                    'toss', 'guess', 'space', 'aspect', 'crass', 'someone', 'school', 'witless', 'excuse', 'meander',
                    'equally', 'merely', 'animated', 'hear', 'person', 'scene', 'suck', 'cheesy', 'creativity',
                    'protagonist', 'artificial', 'rest', 'term', 'ago', 'throughout', 'scifi', 'familiar', 'technical',
                    'manner', 'amateurish', 'brown', 'book', 'rhythm', 'exactly', 'mak', 'cut', 'artist', 'image',
                    'edge', 'question', 'director', 'ive', 'back', 'th', 'chemistry', 'bill', 'lee',
                    'audiences', 'producers', 'spielberg', 'mindnumbingly', 'bond', 'document', 'strike', 'ii',
                    'washington', 'poetry', 'soar', 'vital', 'frame', 'remake', 'filmed', 'ensemble', 'watchable',
                    'wellmade', 'sign', 'wo', 'sound', 'form', 'shoot', 'inept', 'melodrama', 'view',
                    'animation', 'review', 'song', 'musical', 'songs', 'black', 'though', 'thriller', 'theater',
                    'br', 'dialog', 'james', 'one', 'clarity', 'whimsical', 'odd', 'tear', 'guarantee',
                    'journey', 'dead', 'stunt', 'maker', 'novel', 'choose', 'lift', 'period', 'sensual', 'portrayal',
                    'thumb', 'british', 'body', 'quickly', 'characterization', 'worthwhile', 'hand', 'storyline',
                    'first', 'four', 'five', 'ten', 'master', 'wait', 'tone', 'gore', 'ru', 'filmbr',
                    'viewers', 'dr', 'debut', 'predictable', 'fit', 'drug', 'goodhearted', 'jackson', 'reality',
                    'tired', 'career', 'chase', 'suspenseful', 'usual', 'consistently', 'derivative', 'dub', 'steven',
                    'transform', 'culture', 'release', 'death', 'money', 'spy', 'insight', 'motion', 'begin', 'era',
                    'stefan', 'jrs', 'mar', 'palestinian', 'gandolfini', 'elisha', 'taboos',
                    'jessie', 'hindu', 'let', 'gag', 'heartwarming', 'carry', 'value', 'thoughtprovoking', 'gandolfini',
                    'gandolfini', 'theme', 'job', 'die', 'relationship', 'core', 'past', 'modern', 'watch', 'heartfelt',
                    'version', 'live', 'pay', 'water', 'de', 'mov', 'moviebr', 'robert', 'v',
                    'liv', 'david', 'w', 'la', 'sitt', 'volv', 'tak', 'michael', 'ope', 'mr', 'hour', 'movie',
                    'music', 'film', 'two', 'nt', 'th', 'rrb', 'lrb', 'youre', 'john', 'bor', 'ett', 'hollywood',
                    'drama', 'theyre', 'yeah', 'com', 'itbr', 'genre']

word_vectors = KeyedVectors.load_word2vec_format(
        '/home/john/geek_stuff/Data_Set/Google_News_corpus/GoogleNews-vectors-negative300.bin', binary=True, limit=None)

def get_wordnet_pos(treebank_tag):

    ## Removed Adjective from pos tagging as word_lemmatizer convert superlative degree to original form
    # if treebank_tag.startswith('J'):
    #     return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''

def predict_sentiment(model, clean_string_list, start=0):

    ##### Code for With out using Word2vec

    vocab_to_index, index_to_vocab, vocab_frequency_tuple = load_vocab()
    # Model Parameters
    max_word = 15
    max_features = 25000
    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    frequent_words = [val for val, freq in vocab_frequency_tuple][:max_features]
    word_lemmatizer = WordNetLemmatizer()
    input_data = np.zeros([len(clean_string_list), max_word], dtype=np.float64)
    for idx, each_string in enumerate(clean_string_list):
        clean_string = re.sub('[^A-Za-z ]+', '', each_string)
        truncated_data = []
        for word, typ in nltk.pos_tag(word_tokenize(clean_string)):
            typ = get_wordnet_pos(typ)
            if typ:
                lemmatized_word = word_lemmatizer.lemmatize(word, typ).lower()
            else:
                lemmatized_word = word_lemmatizer.lemmatize(word).lower()

            if lemmatized_word in frequent_words:
                truncated_data.append(vocab_to_index[lemmatized_word])
                if len(truncated_data) >= max_word:
                    break

        if len(truncated_data) < max_word:
            truncated_data += [0] * (max_word - len(truncated_data))
        input_data[idx] = truncated_data
    score = model.predict(input_data)

    stri = ''
    for idx in input_data[0]:
        if idx == 0:
            continue
        stri += ' ' + index_to_vocab[idx]

    # print('With a score of -Ve: {0}% +Ve: {1}%'.format(int(score[0][0]*100), int(score[0][1]*100)))
    # if abs(score[0][0] - score[0][1]) <= 0.15:
    #     print('Detected Sentiment Neutral')
    # else:
    #     prediction = np.argmax(score, axis=1)
    #     if prediction == 0:
    #         print('Sentiment Detected Negative')
    #     elif prediction == 1:
    #         print('Sentiment Detected Positive')
    # print('Words Used for Prediction:  {0}'.format(stri))

    #### With googles Word2vec
    max_word = 15

    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    word_lemmatizer = WordNetLemmatizer()
    clean_string = re.sub('[^A-Za-z ]+', '', clean_string)

    input_data = np.zeros([1, max_word, 300], dtype=np.float64)
    idx = 0
    stri = ''
    for word, typ in nltk.pos_tag(word_tokenize(clean_string)):
        if word in stop_words+movie_stop_words or typ in ['NNP', 'NNS']:
            # Removing all stop words and Noun phrases from testing
            print('Word in stop word or noun {0}=== type {1}'.format(word, typ))
            continue
        typ = get_wordnet_pos(typ)
        if typ:
            lemmatized_word = word_lemmatizer.lemmatize(word, typ).lower()
        else:
            lemmatized_word = word_lemmatizer.lemmatize(word).lower()

        try:
            input_data[0, idx] = word_vectors[lemmatized_word]
            idx += 1
            stri += ' ' + lemmatized_word
        except Exception as exc:
            print('word not in vec dict ', lemmatized_word)
            continue

        if idx >= max_word:
            break

    score = model.predict(input_data)
    print('Prediction ', score)

    print('With a score of -Ve: {0}% +Ve: {1}%'.format(int(score[0][0]*100), int(score[0][1]*100)))
    if abs(score[0][0] - score[0][1]) <= 0.15:
        print('Detected Sentiment Neutral')
    else:
        prediction = np.argmax(score, axis=1)
        if prediction == 0:
            print('Sentiment Detected Negative')
        elif prediction == 1:
            print('Sentiment Detected Positive')
    print('Words Used for Prediction:  {0}'.format(stri))

    return score

def attach_model():
    # load json and create model
    with open('/home/janmejaya/sentiment_files/Model_and_data/complete_sentiment_15_word_new.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/john/sentiment_files/model/complete_pre_trained.h5")
    print("Loaded model from disk")

    return loaded_model

def load_vocab():

    with open('/home/janmejaya/sentiment_files/Model_and_data/complete_vocab_15_word.pkl', 'rb') as f:
        data = pickle.load(f)
    vocab_to_index = data['vocab_to_index']
    index_to_vocab = data['index_to_vocab']
    vocab_frequency = data['vocab_frequency_tuple']
    # print('Len of vocab frequency ', len(vocab_frequency))

    return (vocab_to_index, index_to_vocab, vocab_frequency)


if __name__ == '__main__':
    model = attach_model()
    while True:
        data = input('Provide sentence for sentiment Prediction:\n')
        if data:
            start = time.time()
            predict_sentiment(model=model, clean_string_list=[data], start=start)
            print('Time taken {0}'.format(time.time() - start))
        else:
            break