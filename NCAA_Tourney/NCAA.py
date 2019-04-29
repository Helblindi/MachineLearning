import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import math
import csv
import numpy as np

# This one contains stats for every single regular season game played between 1985 and 2015. It mainly
# contains info on the score of the game, the IDs for each team, and where the game was played.
reg_season_compact_pd = pd.read_csv('Data/RegularSeasonCompactResults.csv')
# reg_season_compact_pd.head()

# This one expands on the previous data frame by going into more in depth stats like 3 point field goals,
# free throws, steals, blocks, etc.
reg_season_detailed_pd = pd.read_csv('Data/RegularSeasonDetailedResults.csv')
# reg_season_detailed_pd.columns

# Don't think this data is honestly that important. Just contains the region areas for the tournament each
# year. There isn't really a distinct "home field" advantage in the tourney because the games are supposed
# to be on neutral sites.
seasons_pd = pd.read_csv('Data/Seasons.csv')
# seasons_pd.head()

teams_pd = pd.read_csv('Data/Teams.csv')
teamList = teams_pd['Team_Name'].tolist()
print('size: ', len(teamList))
# teams_pd.tail()

# This one contains the stats for every single NCAA tournament game from 1985 to 2015
tourney_compact_pd = pd.read_csv('Data/TourneyCompactResults.csv')
# tourney_compact_pd.head()

# More deatiled tourney stats (except only stats from 2003 :( )
tourney_detailed_pd = pd.read_csv('Data/TourneyDetailedResults.csv')
# tourney_detailed_pd.head()

# This one tells you what seed each team was for a given tournament year
tourney_seeds_pd = pd.read_csv('Data/TourneySeeds.csv')
# tourney_seeds_pd.head()

# Seeing what seed Duke was in every tourney
duke_id = teams_pd[teams_pd['Team_Name'] == 'Duke'].values[0][0]
# tourney_seeds_pd[tourney_seeds_pd['TeamID'] == duke_id]

# Don't know how helpful this is tbh, because it just tells you what the seeds of the stronger
# and weaker seeds are (assuming that the favored team wins??), so its always 1 vs 16 and then
# 1 vs 8 and 1 vs 4..
tourney_slots_pd = pd.read_csv('Data/TourneySlots.csv')
# tourney_slots_pd.head()

conference_pd = pd.read_csv('Data/Conference.csv')
# conference_pd.head()

tourney_results_pd = pd.read_csv('Data/TourneyResults.csv')
# tourney_results_pd.head()
NCAAChampionsList = tourney_results_pd['NCAA Champion'].tolist()

listACCteams = ['North Carolina', 'Virginia', 'Florida St', 'Louisville', 'Notre Dame', 'Syracuse', 'Duke', 'Virginia Tech', 'Georgia Tech', 'Miami', 'Wake Forest', 'Clemson', 'NC State', 'Boston College', 'Pittsburgh']
listPac12teams = ['Arizona', 'Oregon', 'UCLA', 'California', 'USC', 'Utah', 'Washington St', 'Stanford', 'Arizona St', 'Colorado', 'Washington', 'Oregon St']
listSECteams = ['Kentucky', 'South Carolina', 'Florida', 'Arkansas', 'Alabama', 'Tennessee', 'Mississippi St', 'Georgia', 'Ole Miss', 'Vanderbilt', 'Auburn', 'Texas A&M', 'LSU', 'Missouri']
listBig10teams = ['Maryland', 'Wisconsin', 'Purdue', 'Northwestern', 'Michigan St', 'Indiana', 'Iowa', 'Michigan', 'Penn St', 'Nebraska', 'Minnesota', 'Illinois', 'Ohio St', 'Rutgers']
listBig12teams = ['Kansas', 'Baylor', 'West Virginia', 'Iowa St', 'TCU', 'Kansas St', 'Texas Tech', 'Oklahoma St', 'Texas', 'Oklahoma']
listBigEastteams = ['Butler', 'Creighton', 'DePaul', 'Georgetown', 'Marquette', 'Providence', 'Seton Hall', 'St John\'s', 'Villanova', 'Xavier']


def checkPower6Conference(team_id):
    teamName = teams_pd.values[team_id-1101][1]
    if (teamName in listACCteams or teamName in listBig10teams or teamName in listBig12teams
       or teamName in listSECteams or teamName in listPac12teams or teamName in listBigEastteams):
        return 1
    else:
        return 0


def getTeamID(name):
    return teams_pd[teams_pd['Team_Name'] == name].values[0][0]


def getTeamName(team_id):
    return teams_pd[teams_pd['Team_Id'] == team_id].values[0][1]


def getNumChampionships(team_id):
    name = getTeamName(team_id)
    return NCAAChampionsList.count(name)


def getListForURL(team_list):
    team_list = [x.lower() for x in team_list]
    team_list = [t.replace(' ', '-') for t in team_list]
    team_list = [t.replace('st', 'state') for t in team_list]
    team_list = [t.replace('northern-dakota', 'north-dakota') for t in team_list]
    team_list = [t.replace('nc-', 'north-carolina-') for t in team_list]
    team_list = [t.replace('fl-', 'florida-') for t in team_list]
    team_list = [t.replace('ga-', 'georgia-') for t in team_list]
    team_list = [t.replace('lsu', 'louisiana-state') for t in team_list]
    team_list = [t.replace('maristate', 'marist') for t in team_list]
    team_list = [t.replace('stateate', 'state') for t in team_list]
    team_list = [t.replace('northernorthern', 'northern') for t in team_list]
    team_list = [t.replace('usc', 'southern-california') for t in team_list]
    base = 'http://www.sports-reference.com/cbb/schools/'
    for team in team_list:
        url = base + team + '/'


# getListForURL(teamList)


# Function for handling the annoying cases of Florida and FL, as well as State and St
def handleCases(arr):
    indices = []
    listLen = len(arr)
    for i in range(listLen):
        if arr[i] == 'St' or arr[i] == 'FL':
            indices.append(i)
    for p in indices:
        arr[p-1] = arr[p-1] + ' ' + arr[p]
    for i in range(len(indices)):
        arr.remove(arr[indices[i] - i])
    return arr


def checkConferenceChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Regular Season Champ'].tolist()
    # For handling cases where there is more than one champion
    champs_separated = [words for segments in champs for words in segments.split()]
    name = getTeamName(team_id)
    champs_separated = handleCases(champs_separated)
    if name in champs_separated:
        return 1
    else:
        return 0


def checkConferenceTourneyChamp(team_id, year):
    year_conf_pd = conference_pd[conference_pd['Year'] == year]
    champs = year_conf_pd['Tournament Champ'].tolist()
    name = getTeamName(team_id)
    if name in champs:
        return 1
    else:
        return 0


def getTourneyAppearances(team_id):
    return len(tourney_seeds_pd[tourney_seeds_pd['TeamID'] == team_id].index)


def handleDifferentCSV(df):
    # The stats CSV is a lit different in terms of naming so below is just some data cleaning
    df['School'] = df['School'].replace('(State)', 'St', regex=True)
    df['School'] = df['School'].replace('Albany (NY)', 'Albany NY')
    df['School'] = df['School'].replace('Boston University', 'Boston Univ')
    df['School'] = df['School'].replace('Central Michigan', 'C Michigan')
    df['School'] = df['School'].replace('(Eastern)', 'E', regex=True)
    df['School'] = df['School'].replace('Louisiana St', 'LSU')
    df['School'] = df['School'].replace('North Carolina St', 'NC State')
    df['School'] = df['School'].replace('Southern California', 'USC')
    df['School'] = df['School'].replace('University of California', 'California', regex=True)
    df['School'] = df['School'].replace('American', 'American Univ')
    df['School'] = df['School'].replace('Arkansas-Little Rock', 'Ark Little Rock')
    df['School'] = df['School'].replace('Arkansas-Pine Bluff', 'Ark Pine Bluff')
    df['School'] = df['School'].replace('Bowling Green St', 'Bowling Green')
    df['School'] = df['School'].replace('Brigham Young', 'BYU')
    df['School'] = df['School'].replace('Cal Poly', 'Cal Poly SLO')
    df['School'] = df['School'].replace('Centenary (LA)', 'Centenary')
    df['School'] = df['School'].replace('Central Connecticut St', 'Central Conn')
    df['School'] = df['School'].replace('Charleston Southern', 'Charleston So')
    df['School'] = df['School'].replace('Coastal Carolina', 'Coastal Car')
    df['School'] = df['School'].replace('College of Charleston', 'Col Charleston')
    df['School'] = df['School'].replace('Cal St Fullerton', 'CS Fullerton')
    df['School'] = df['School'].replace('Cal St Sacramento', 'CS Sacramento')
    df['School'] = df['School'].replace('Cal St Bakersfield', 'CS Bakersfield')
    df['School'] = df['School'].replace('Cal St Northridge', 'CS Northridge')
    df['School'] = df['School'].replace('East Tennessee St', 'ETSU')
    df['School'] = df['School'].replace('Detroit Mercy', 'Detroit')
    df['School'] = df['School'].replace('Fairleigh Dickinson', 'F Dickinson')
    df['School'] = df['School'].replace('Florida Atlantic', 'FL Atlantic')
    df['School'] = df['School'].replace('Florida Gulf Coast', 'FL Gulf Coast')
    df['School'] = df['School'].replace('Florida International', 'Florida Intl')
    df['School'] = df['School'].replace('George Washington', 'G Washington')
    df['School'] = df['School'].replace('Georgia Southern', 'Ga Southern')
    df['School'] = df['School'].replace('Gardner-Webb', 'Gardner Webb')
    df['School'] = df['School'].replace('Illinois-Chicago', 'IL Chicago')
    df['School'] = df['School'].replace('Kent St', 'Kent')
    df['School'] = df['School'].replace('Long Island University', 'Long Island')
    df['School'] = df['School'].replace('Loyola Marymount', 'Loy Marymount')
    df['School'] = df['School'].replace('Loyola (MD)', 'Loyola MD')
    df['School'] = df['School'].replace('Loyola (IL)', 'Loyola-Chicago')
    df['School'] = df['School'].replace('Massachusetts', 'MA Lowell')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    df['School'] = df['School'].replace('Miami (FL)', 'Miami FL')
    df['School'] = df['School'].replace('Miami (OH)', 'Miami OH')
    df['School'] = df['School'].replace('Missouri-Kansas City', 'Missouri KC')
    df['School'] = df['School'].replace('Monmouth', 'Monmouth NJ')
    df['School'] = df['School'].replace('Mississippi Valley St', 'MS Valley St')
    df['School'] = df['School'].replace('Montana St', 'MTSU')
    df['School'] = df['School'].replace('Northern Colorado', 'N Colorado')
    df['School'] = df['School'].replace('North Dakota St', 'N Dakota St')
    df['School'] = df['School'].replace('Northern Illinois', 'N Illinois')
    df['School'] = df['School'].replace('Northern Kentucky', 'N Kentucky')
    df['School'] = df['School'].replace('North Carolina A&T', 'NC A&T')
    df['School'] = df['School'].replace('North Carolina Central', 'NC Central')
    df['School'] = df['School'].replace('Pennsylvania', 'Penn')
    df['School'] = df['School'].replace('South Carolina St', 'S Carolina St')
    df['School'] = df['School'].replace('Southern Illinois', 'S Illinois')
    df['School'] = df['School'].replace('UC-Santa Barbara', 'Santa Barbara')
    df['School'] = df['School'].replace('Southeastern Louisiana', 'SE Louisiana')
    df['School'] = df['School'].replace('Southeast Missouri St', 'SE Missouri St')
    df['School'] = df['School'].replace('Stephen F. Austin', 'SF Austin')
    df['School'] = df['School'].replace('Southern Methodist', 'SMU')
    df['School'] = df['School'].replace('Southern Mississippi', 'Southern Miss')
    df['School'] = df['School'].replace('Southern', 'Southern Univ')
    df['School'] = df['School'].replace('St. Bonaventure', 'St Bonaventure')
    df['School'] = df['School'].replace('St. Francis (NY)', 'St Francis NY')
    df['School'] = df['School'].replace('Saint Francis (PA)', 'St Francis PA')
    df['School'] = df['School'].replace('St. John\'s (NY)', 'St John\'s')
    df['School'] = df['School'].replace('Saint Joseph\'s', 'St Joseph\'s PA')
    df['School'] = df['School'].replace('Saint Louis', 'St Louis')
    df['School'] = df['School'].replace('Saint Mary\'s (CA)', 'St Mary\'s CA')
    df['School'] = df['School'].replace('Mount Saint Mary\'s', 'Mt St Mary\'s')
    df['School'] = df['School'].replace('Saint Peter\'s', 'St Peter\'s')
    df['School'] = df['School'].replace('Texas A&M-Corpus Christian', 'TAM C. Christian')
    df['School'] = df['School'].replace('Texas Christian', 'TCU')
    df['School'] = df['School'].replace('Tennessee-Martin', 'TN Martin')
    df['School'] = df['School'].replace('Texas-Rio Grande Valley', 'UTRGV')
    df['School'] = df['School'].replace('Texas Southern', 'TX Southern')
    df['School'] = df['School'].replace('Alabama-Birmingham', 'UAB')
    df['School'] = df['School'].replace('UC-Davis', 'UC Davis')
    df['School'] = df['School'].replace('UC-Irvine', 'UC Irvine')
    df['School'] = df['School'].replace('UC-Riverside', 'UC Riverside')
    df['School'] = df['School'].replace('Central Florida', 'UCF')
    df['School'] = df['School'].replace('Louisiana-Lafayette', 'ULL')
    df['School'] = df['School'].replace('Louisiana-Monroe', 'ULM')
    df['School'] = df['School'].replace('Maryland-Baltimore County', 'UMBC')
    df['School'] = df['School'].replace('North Carolina-Asheville', 'UNC Asheville')
    df['School'] = df['School'].replace('North Carolina-Greensboro', 'UNC Greensboro')
    df['School'] = df['School'].replace('North Carolina-Wilmington', 'UNC Wilmington')
    df['School'] = df['School'].replace('Nevada-Las Vegas', 'UNLV')
    df['School'] = df['School'].replace('Texas-Arlington', 'UT Arlington')
    df['School'] = df['School'].replace('Texas-San Antonio', 'UT San Antonio')
    df['School'] = df['School'].replace('Texas-El Paso', 'UTEP')
    df['School'] = df['School'].replace('Virginia Commonwealth', 'VA Commonwealth')
    df['School'] = df['School'].replace('Western Carolina', 'W Carolina')
    df['School'] = df['School'].replace('Western Illinois', 'W Illinois')
    df['School'] = df['School'].replace('Western Kentucky', 'WKU')
    df['School'] = df['School'].replace('Western Michigan', 'W Michigan')
    df['School'] = df['School'].replace('Abilene Christian', 'Abilene Chr')
    df['School'] = df['School'].replace('Montana State', 'Montana St')
    df['School'] = df['School'].replace('Central Arkansas', 'Cent Arkansas')
    df['School'] = df['School'].replace('Houston Baptist', 'Houston Bap')
    df['School'] = df['School'].replace('South Dakota St', 'S Dakota St')
    df['School'] = df['School'].replace('Maryland-Eastern Shore', 'MD E Shore')
    return df


def getSeasonData(team_id, year):
    # The data frame below holds stats for every single game in the given year
    year_data_pd = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
    # print('\n HERE IT IS!!')
    # print(year_data_pd)
    # Finding number of points per game
    # print('type 1: ', type(year_data_pd.WTeamID))
    # print('type 2: ', type(team_id))
    gamesWon = year_data_pd[year_data_pd.WTeamID == team_id]
    totalPointsScored = gamesWon['WScore'].sum()
    gamesLost = year_data_pd[year_data_pd.LTeamID == team_id]
    totalGames = gamesWon.append(gamesLost)
    numGames = len(totalGames.index)
    totalPointsScored += gamesLost['LScore'].sum()

    # Finding number of points per game allowed
    totalPointsAllowed = gamesWon['LScore'].sum()
    totalPointsAllowed += gamesLost['WScore'].sum()

    y = year
    if year == 2018:
        y = 2017
    stats_SOS_pd = pd.read_csv('Data/MMStats/MMStats_' + str(y) + '.csv')
    stats_SOS_pd = handleDifferentCSV(stats_SOS_pd)
    ratings_pd = pd.read_csv('Data/RatingStats/RatingStats_' + str(y) + '.csv')
    ratings_pd = handleDifferentCSV(ratings_pd)

    name = getTeamName(team_id)
    team = stats_SOS_pd[stats_SOS_pd['School'] == name]
    team_rating = ratings_pd[ratings_pd['School'] == name]
    if len(team.index) == 0 or len(team_rating.index) == 0:  # Can't find the team
        total3sMade = 0
        totalTurnovers = 0
        totalAssists = 0
        sos = 0
        totalRebounds = 0
        srs = 0
        totalSteals = 0
    else:
        total3sMade = team['X3P'].values[0]
        totalTurnovers = team['TOV'].values[0]
        if math.isnan(totalTurnovers):
            totalTurnovers = 0
        totalAssists = team['AST'].values[0]
        if math.isnan(totalAssists):
            totalAssists = 0
        sos = team['SOS'].values[0]
        srs = team['SRS'].values[0]
        totalRebounds = team['TRB'].values[0]
        if math.isnan(totalRebounds):
            totalRebounds = 0
        totalSteals = team['STL'].values[0]
        if math.isnan(totalSteals):
            totalSteals = 0

    # Finding tournament seed for that year
    tourneyYear = tourney_seeds_pd[tourney_seeds_pd['Season'] == year]
    seed = tourneyYear[tourneyYear['TeamID'] == team_id]
    if len(seed.index) != 0:
        seed = seed.values[0][1]
        tournamentSeed = int(seed[1:3])
    else:
        tournamentSeed = 25  # Not sure how to represent if a team didn't make the tourney

    # Finding number of wins and losses
    numWins = len(gamesWon.index)
    # There are some teams who may have dropped to Division 2, so they won't have games
    # a certain year. In this case, we don't want to divide by 0, so we'll just set the
    # averages to 0 instead
    if numGames == 0:
        avgPointsScored = 0
        avgPointsAllowed = 0
        avg3sMade = 0
        avgTurnovers = 0
        avgAssists = 0
        avgRebounds = 0
        avgSteals = 0
    else:
        avgPointsScored = totalPointsScored / numGames
        avgPointsAllowed = totalPointsAllowed / numGames
        avg3sMade = total3sMade / numGames
        avgTurnovers = totalTurnovers / numGames
        avgAssists = totalAssists / numGames
        avgRebounds = totalRebounds / numGames
        avgSteals = totalSteals / numGames
    # return [numWins, sos, srs]
    # return [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avg3sMade, avg3sAllowed, avgTurnovers,
    #        tournamentSeed, getStrengthOfSchedule(team_id, year), getTourneyAppearances(team_id)]
    return [numWins, avgPointsScored, avgPointsAllowed, checkPower6Conference(team_id), avg3sMade, avgAssists,
            avgTurnovers,
            checkConferenceChamp(team_id, year), checkConferenceTourneyChamp(team_id, year), tournamentSeed,
            sos, srs, avgRebounds, avgSteals, getTourneyAppearances(team_id), getNumChampionships(team_id)]


# kentucky_id = teams_pd[teams_pd['Team_Name'] == 'Kentucky'].values[0][0]
# getSeasonData(kentucky_id, 2016)

def compareTwoTeams(id_1, id_2, year):
    team_1 = getSeasonData(id_1, year)
    team_2 = getSeasonData(id_2, year)
    diff = [a - b for a, b in zip(team_1, team_2)]
    return diff


# kansas_id = teams_pd[teams_pd['Team_Name'] == 'Kansas'].values[0][0]
# compareTwoTeams(kansas_id, kentucky_id, 2016)

def createSeasonDict(year):
    seasonDictionary = collections.defaultdict(list)
    for team in teamList:
        team_id = teams_pd[teams_pd['Team_Name'] == team].values[0][0]
        team_vector = getSeasonData(team_id, year)
        seasonDictionary[team_id] = team_vector
    return seasonDictionary


def getHomeStat(row):
    if row == 'H':
        home = 1
    if row == 'A':
        home = -1
    if row == 'N':
        home = 0
    return home


def createTrainingSet(years):
    totalNumGames = 0
    for year in years:
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        totalNumGames += len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        totalNumGames += len(tourney.index)
    numFeatures = len(getSeasonData(1181,2012)) #Just choosing a random team and seeing the dimensionality of the vector
    xTrain = np.zeros(( totalNumGames, numFeatures + 1))
    yTrain = np.zeros(totalNumGames)
    indexCounter = 0
    for year in years:
        team_vectors = createSeasonDict(year)
        season = reg_season_compact_pd[reg_season_compact_pd['Season'] == year]
        numGamesInSeason = len(season.index)
        tourney = tourney_compact_pd[tourney_compact_pd['Season'] == year]
        numGamesInSeason += len(tourney.index)
        xTrainSeason = np.zeros(( numGamesInSeason, numFeatures + 1))
        yTrainSeason = np.zeros(( numGamesInSeason ))
        counter = 0
        for index, row in season.iterrows():
            w_team = row['Wteam']
            w_vector = team_vectors[w_team]
            l_team = row['Lteam']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = getHomeStat(row['Wloc'])
            if counter % 2 == 0:
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [ -p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        for index, row in tourney.iterrows():
            w_team = row['Wteam']
            w_vector = team_vectors[w_team]
            l_team = row['Lteam']
            l_vector = team_vectors[l_team]
            diff = [a - b for a, b in zip(w_vector, l_vector)]
            home = 0 #All tournament games are neutral
            if counter % 2 == 0:
                diff.append(home)
                xTrainSeason[counter] = diff
                yTrainSeason[counter] = 1
            else:
                diff.append(-home)
                xTrainSeason[counter] = [ -p for p in diff]
                yTrainSeason[counter] = 0
            counter += 1
        xTrain[indexCounter:numGamesInSeason+indexCounter] = xTrainSeason
        yTrain[indexCounter:numGamesInSeason+indexCounter] = yTrainSeason
        indexCounter += numGamesInSeason
    return xTrain, yTrain


def normalizeInput(arr):
    for i in range(arr.shape[1]):
        minVal = min(arr[:, i])
        maxVal = max(arr[:, i])
        arr[:, i] = (arr[:, i] - minVal) / (maxVal - minVal)
    return arr


# alternative:
def normalize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)



years = range(1993, 2018)
# xTrain, yTrain = createTrainingSet(years)
# np.save('xTrain', xTrain)
# np.save('yTrain', yTrain)

xTrain = np.load('PrecomputedMatrices/xTrain.npy')
yTrain = np.load('PrecomputedMatrices/yTrain.npy')

# xTrain.shape
# yTrain.shape

# These are the different models I tried. Simply uncomment the model that you want to try.

# model = tree.DecisionTreeClassifier()
# model = tree.DecisionTreeRegressor()
# model = linear_model.LogisticRegression()
# model = linear_model.BayesianRidge()
# model = linear_model.Lasso()
# model = svm.SVC()
# model = svm.SVR()
# model = linear_model.Ridge(alpha = 0.5)
# model = AdaBoostClassifier(n_estimators=100)
# model = GradientBoostingClassifier(n_estimators=100)
# model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
# model = RandomForestClassifier(n_estimators=64)
model = KNeighborsClassifier(n_neighbors=39)
# neuralNetwork(10)
# model = VotingClassifier(estimators=[('GBR', model1), ('BR', model2), ('KNN', model3)], voting='soft')
# model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=0.1)


def showDependency(predictions, test, stat, my_categories):
    difference = test[:,my_categories.index(stat)]
    plt.scatter(difference, predictions)
    plt.ylabel('Probability of Team 1 Win')
    plt.xlabel(stat + ' Difference (Team 1 - Team 2)')
    plt.show()


def showFeatureImportance(my_categories):
    fx_imp = pd.Series(model.feature_importances_, index=my_categories)
    fx_imp /= fx_imp.max()
    fx_imp.sort()
    fx_imp.plot(kind='barh')


categories = ['Wins', 'PPG', 'PPGA', 'PowerConf', '3PG', 'APG', 'TOP', 'Conference Champ', 'Tourney Conference Champ',
              'Seed', 'SOS', 'SRS', 'RPG', 'SPG', 'Tourney Appearances', 'National Championships', 'Location']
accuracy = []

for i in range(1):
    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
    results = model.fit(X_train, Y_train)
    preds = model.predict(X_test)

    preds[preds < .5] = 0
    preds[preds >= .5] = 1
    accuracy.append(np.mean(preds == Y_test))
    # accuracy.append(np.mean(predictions == Y_test))
    print("Finished iteration:", i)

print("The accuracy is: ", sum(accuracy)/len(accuracy))

# categories=['Wins', 'PPG', 'PPGA', 'PowerConf', '3PG', 'APG', 'TOP', 'Conference Champ', 'Tourney Conference Champ',
#           'Seed', 'SOS', 'SRS', 'RPG', 'SPG', 'Tourney Appearances', 'National Championships', 'Location']
# accuracy=[]

# for i in range(1):
#    X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
#    results1 = model.fit(X_train, Y_train)
#    preds1 = model.predict(X_test)
#
#    results2 = model2.fit(X_train, Y_train)
#    preds2 = model2.predict(X_test)

#    results3 = model3.fit(X_train, Y_train)
#    preds3 = model3.predict(X_test)

#    results4 = model4.fit(X_train, Y_train)
#    preds4 = model4.predict(X_test)

#    results5 = model5.fit(X_train, Y_train)
#    preds5 = model5.predict(X_test)

#    preds = (preds1 + preds2 + preds3 + preds4 + preds5)/5

#    preds[preds < .5] = 0
#    preds[preds >= .5] = 1
#    accuracy.append(np.mean(preds == Y_test))
# accuracy.append(np.mean(predictions == Y_test))
# print "The accuracy is", sum(accuracy)/len(accuracy)
# showFeatureImportance(categories)


def predictGame(team_1_vector, team_2_vector, home):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    return model.predict([diff])
    # return model.predict_proba([diff])


# This was the national championship matchup last year
team1_name = 'North Carolina'
team2_name = 'Villanova'
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
print('Probability that ' + team1_name + ' wins:', predictGame(team1_vector, team2_vector, 0)[0])

sample_sub_pd = pd.read_csv('Data/sample_submission.csv')
(sample_sub_pd.head())

def createPrediction():
    results = [[0 for x in range(2)] for x in range(len(sample_sub_pd.index))]
    for index, row in sample_sub_pd.iterrows():
        matchup_id = row['id']
        year = matchup_id[0:4]
        team1_id = matchup_id[5:9]
        team2_id = matchup_id[10:14]
        team1_vector = getSeasonData(int(team1_id), int(year))
        team2_vector = getSeasonData(int(team2_id), int(year))
        pred = predictGame(team1_vector, team2_vector, 0)
        results[index][0] = matchup_id
        results[index][1] = pred[0]
        #results[index][1] = pred[0][1]
    results = pd.np.array(results)
    firstRow = [[0 for x in range(2)] for x in range(1)]
    firstRow[0][0] = 'id'
    firstRow[0][1] = 'pred'
    with open("result.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(firstRow)
        writer.writerows(results)


# createPrediction()


def findBestK():
    K = list(i for i in range(1, 200) if i % 2 != 0)
    p = []
    for k in K:
        kmeans = KNeighborsClassifier(n_neighbors=k)
        kmeans.fit(X_train, Y_train)
        results = kmeans.fit(X_train, Y_train)
        preds = kmeans.predict(X_test)
        p.append(np.mean(preds == Y_test))
    plt.plot(K, p)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Selecting k with the Elbow Method')
    plt.show()


# def neuralNetwork(loops):
#     accuracy=[]
#     dim = 17
#     for i in range(loops):
#         X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain)
#
#         model = Sequential()
#         model.add(Convolution1D(28, 3, border_mode='same', init='normal', input_shape=(dim, 1))) #28 1D filters of length 3
#         model.add(Dense(128, init='normal', input_dim = dim))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.2))
#         model.add(Dense(64, init='normal'))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.1))
#         model.add(Dense(16, init='normal'))
#         model.add(Activation('relu'))
#         model.add(Dense(2, init='normal'))
#         model.add(Activation('softmax'))
#
#
#         #X_train = X_train.reshape((len(X_train), dim, 1))
#         #X_test = X_test.reshape((len(X_test), dim, 1))
#         Y_train_categorical = np_utils.to_categorical(Y_train)
#         # TRAINING
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         model.fit(X_train, Y_train_categorical, batch_size=64, nb_epoch=10,shuffle=True)
#         preds = model.predict(X_test)
#         results=[]
#         for i in range(preds.shape[0]):
#             if preds[i][1] < .5:
#                 results.append(0)
#             else:
#                 results.append(1)
#         accuracy.append(np.mean(results == Y_test))
#     #accuracy.append(np.mean(predictions == Y_test))
#     print("The accuracy is", sum(accuracy)/len(accuracy))


def getAllTeamVectors():
    year = 2017
    numFeatures = len(getSeasonData(1181,2012))
    teamvecs = np.zeros((364, numFeatures))
    teams = []
    counter = 0
    for team in teamList:
        team_id = teams_pd[teams_pd['Team_Name'] == team].values[0][0]
        team_vector = getSeasonData(team_id, year)
        if team_vector[0] == 0 or team_vector[4] == 0:
            continue
        teamvecs[counter] = team_vector
        teams.append(team)
        counter += 1
    team = pd.np.array(teams)
    team = np.reshape(team, (team.shape[0], 1))
    with open("allNames.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(team)
    with open("allVecs.tsv", "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(teamvecs)


#print(teamList)

#getAllTeamVectors()

# execute actual analysis for 2018
# Create variable with TRUE if Season in 2018
_2018 = tourney_seeds_pd['Season'] == 2018

# select all cases where Season is 2018
_2018_tourney_seeds = tourney_seeds_pd[_2018]

# Drop unnecessary columns
_2018_tourney_seeds = _2018_tourney_seeds.drop(columns=['Season', 'Seed'])

# convert pandas dataframe to an np array
_2018_tourney_seeds = _2018_tourney_seeds.values

########################################################################################################################
# run the First Four
_round1 = _2018_tourney_seeds

print('starting length: ', len(_round1))

# 10 vs 11
team1_name = getTeamName(int(_2018_tourney_seeds[10]))
team2_name = getTeamName(int(_2018_tourney_seeds[11]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_round1 == getTeamID(team2_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_round1 == getTeamID(team1_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

# 16 vs 17
team1_name = getTeamName(int(_2018_tourney_seeds[16]))
team2_name = getTeamName(int(_2018_tourney_seeds[17]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_round1 == getTeamID(team2_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_round1 == getTeamID(team1_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

# 28 vs 29
team1_name = getTeamName(int(_2018_tourney_seeds[28]))
team2_name = getTeamName(int(_2018_tourney_seeds[29]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_round1 == getTeamID(team2_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_round1 == getTeamID(team1_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

# 66 vs 67
team1_name = getTeamName(int(_2018_tourney_seeds[66]))
team2_name = getTeamName(int(_2018_tourney_seeds[67]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_round1 == getTeamID(team2_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_round1 == getTeamID(team1_name))
    _round1 = np.delete(_round1, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

# account for error (not sure why the 1st element is getting removed
array = np.array([1437])
_round1 = np.append(array, _round1)

print('length after first four: ', len(_round1))

########################################################################################################################
# Round 1
_round2 = _round1
print('starting length: ', len(_round2))

print('\nEast Conference 1st Round')
for it in range(0, 8):
    team1_name = getTeamName(int(_round1[it]))
    team2_name = getTeamName(int(_round1[15 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_round2 == getTeamID(team2_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_round2 == getTeamID(team1_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nMidwest Conference 1st Round')
for it in range(0, 8):
    team1_name = getTeamName(int(_round1[16 + it]))
    team2_name = getTeamName(int(_round1[31 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_round2 == getTeamID(team2_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_round2 == getTeamID(team1_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nSouth Conference 1st Round')
for it in range(0, 8):
    team1_name = getTeamName(int(_round1[32 + it]))
    team2_name = getTeamName(int(_round1[47 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_round2 == getTeamID(team2_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_round2 == getTeamID(team1_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nWest Conference 1st Round')
for it in range(0, 8):
    team1_name = getTeamName(int(_round1[48 + it]))
    team2_name = getTeamName(int(_round1[63 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_round2 == getTeamID(team2_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_round2 == getTeamID(team1_name))
        _round2 = np.delete(_round2, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('ending length: ', len(_round2))

########################################################################################################################
# Round 2
_sweet_sixteen = _round2
print('starting length: ', len(_sweet_sixteen))

print('\nEast Conference 2nd Round')
for it in range(0, 4):
    team1_name = getTeamName(int(_round2[it]))
    team2_name = getTeamName(int(_round2[7 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_sweet_sixteen == getTeamID(team2_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_sweet_sixteen == getTeamID(team1_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nMidwest Conference 2nd Round')
for it in range(0, 4):
    team1_name = getTeamName(int(_round2[8 +it]))
    team2_name = getTeamName(int(_round2[15 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_sweet_sixteen == getTeamID(team2_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_sweet_sixteen == getTeamID(team1_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nSouth Conference 2nd Round')
for it in range(0, 4):
    team1_name = getTeamName(int(_round2[16 + it]))
    team2_name = getTeamName(int(_round2[23 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_sweet_sixteen == getTeamID(team2_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_sweet_sixteen == getTeamID(team1_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nWest Conference 2nd Round')
for it in range(0, 4):
    team1_name = getTeamName(int(_round2[24 + it]))
    team2_name = getTeamName(int(_round2[31 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_sweet_sixteen == getTeamID(team2_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_sweet_sixteen == getTeamID(team1_name))
        _sweet_sixteen = np.delete(_sweet_sixteen, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('ending len: ', len(_sweet_sixteen))

########################################################################################################################
# Sweet 16
_elite_eight = _sweet_sixteen
print('starting length: ', len(_elite_eight))

print('\nEast Conference Sweet Sixteen')
for it in range(0, 2):
    team1_name = getTeamName(int(_sweet_sixteen[it]))
    team2_name = getTeamName(int(_sweet_sixteen[3 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_elite_eight == getTeamID(team2_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_elite_eight == getTeamID(team1_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nMidwest Conference Sweet Sixteen')
for it in range(0, 2):
    team1_name = getTeamName(int(_sweet_sixteen[4 + it]))
    team2_name = getTeamName(int(_sweet_sixteen[7 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_elite_eight == getTeamID(team2_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_elite_eight == getTeamID(team1_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nSouth Conference Sweet Sixteen')
for it in range(0, 2):
    team1_name = getTeamName(int(_sweet_sixteen[8 + it]))
    team2_name = getTeamName(int(_sweet_sixteen[11 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_elite_eight == getTeamID(team2_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_elite_eight == getTeamID(team1_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nWest Conference Sweet Sixteen')
for it in range(0, 2):
    team1_name = getTeamName(int(_sweet_sixteen[12 + it]))
    team2_name = getTeamName(int(_sweet_sixteen[15 - it]))
    team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
    team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
    if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
        i = np.where(_elite_eight == getTeamID(team2_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
    else:
        i = np.where(_elite_eight == getTeamID(team1_name))
        _elite_eight = np.delete(_elite_eight, i)
        print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('ending len: ', len(_elite_eight))

########################################################################################################################
# Elite 8
_final_four = _elite_eight
print('starting length: ', len(_final_four))

print('\nEast Conference Elite Eight')
team1_name = getTeamName(int(_elite_eight[0]))
team2_name = getTeamName(int(_elite_eight[1]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_final_four == getTeamID(team2_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_final_four == getTeamID(team1_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nMidwest Conference Elite Eight')
team1_name = getTeamName(int(_elite_eight[2]))
team2_name = getTeamName(int(_elite_eight[3]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_final_four == getTeamID(team2_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_final_four == getTeamID(team1_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nSouth Conference Elite Eight')
team1_name = getTeamName(int(_elite_eight[4]))
team2_name = getTeamName(int(_elite_eight[5]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_final_four == getTeamID(team2_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_final_four == getTeamID(team1_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('\nWest Conference Elite Eight')
team1_name = getTeamName(int(_elite_eight[6]))
team2_name = getTeamName(int(_elite_eight[7]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_final_four == getTeamID(team2_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_final_four == getTeamID(team1_name))
    _final_four = np.delete(_final_four, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

print('ending len: ', len(_final_four))

########################################################################################################################
# Final Four
_last_two = _final_four
print('starting length: ', len(_last_two))
team1_name = getTeamName(int(_final_four[0]))
team2_name = getTeamName(int(_final_four[1]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_last_two == getTeamID(team2_name))
    _last_two = np.delete(_last_two, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_last_two == getTeamID(team1_name))
    _last_two = np.delete(_last_two, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

team1_name = getTeamName(int(_final_four[2]))
team2_name = getTeamName(int(_final_four[3]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_last_two == getTeamID(team2_name))
    _last_two = np.delete(_last_two, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team1_name)
else:
    i = np.where(_last_two == getTeamID(team1_name))
    _last_two = np.delete(_last_two, i)
    print('Winner between', team1_name, ' and ', team2_name, ' is: ', team2_name)

########################################################################################################################
# National Championship
print('FINALS')
team1_name = getTeamName(int(_last_two[0]))
team2_name = getTeamName(int(_last_two[1]))
team1_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team1_name].values[0][0], 2016)
team2_vector = getSeasonData(teams_pd[teams_pd['Team_Name'] == team2_name].values[0][0], 2016)
if (predictGame(team1_vector, team2_vector, 0)[0]) >= .5:
    i = np.where(_last_two == getTeamID(team2_name))
    _last_two = np.delete(_last_two, i)
    print('Final Winner: ', team1_name)
else:
    i = np.where(_last_two == getTeamID(team1_name))
    _last_two = np.delete(_last_two, i)
    print('Final Winner: ', team2_name)

