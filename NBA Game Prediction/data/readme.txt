basketballreference.com stats download 1.1
last updated: 9/12/2005

****************************************************************************
All stats contained in this download are free to use under the following
conditions:

1. Please reference basketballreference.com in all places, print or online, 
where the stats are used.

2. Report any errors found in this download to support@basketballreference.com.
If these stats help you out then help others by making this the best 
statistical archive online.
****************************************************************************

All player season stats include the 2004-05 regular season  and playoff stats.  
-NBA Draft is only through the 2004 draft
-All-Star games are only through the 2003-04 season
-Coaching records do not include 2004-05 season


This download contains the following comma delimited files

players.txt - list of all players
player_regular_season.txt - regular season player stats
player_regular_season_career.txt - regular season career totals by player
player_playoffs.txt - playoff stats for all players
player_playoffs_career.txt - career playoff stats by player
player_allstar.txt - all-star stats by player
teams.txt - list of all teams
team_season.txt - regular season team stats
draft.txt - nba and aba draft results by year
coaches_season.txt - nba coaching records by season
coaches_career.txt - nba career coaching records

A few notes about the stats

Steals, blocks and turnovers were not official nba stats until the 70's. Those 
stats are listed as zero for earlier season.

These stats are accurate to the best of our knowledge.  We make no guarantees
about their accuracy and there are still some errors in the data which we are 
trying to fix.  We cannot be held responsible for any damage arising from the 
use of these stats.  Please use at your own risk.

support@basketballreference.com



    team1_stats = df_team_season[df_team_season['team'] == team1].iloc[0]
    team2_stats = df_team_season[df_team_season['team'] == team2].iloc[0]
    team1_stats = team1_stats[['o_field_goals_made','o_field_goals_attempted','o_free_throws_made','o_free_throws_attempted','o_offensive_rebounds','o_defensive_rebounds','o_total_rebounds','o_assists','o_pf','o_steals','o_turnovers','o_blocks','o_3points_made','o_3points_attempted','o_total_points','d_field_goals_made','d_free_goals_attempted','d_free_throws_made','d_freethows_attempted','d_offensive_rebounds','d_defensive_rebounds','d_total_rebounds','d_assists','d_pf','d_steals','d_turnovers','d_blocks','d_3points_made','d_3points_attempted','d_total_points','pace']]
    team2_stats = team2_stats[['o_field_goals_made','o_field_goals_attempted','o_free_throws_made','o_free_throws_attempted','o_offensive_rebounds','o_defensive_rebounds','o_total_rebounds','o_assists','o_pf','o_steals','o_turnovers','o_blocks','o_3points_made','o_3points_attempted','o_total_points','d_field_goals_made','d_free_goals_attempted','d_free_throws_made','d_freethows_attempted','d_offensive_rebounds','d_defensive_rebounds','d_total_rebounds','d_assists','d_pf','d_steals','d_turnovers','d_blocks','d_3points_made','d_3points_attempted','d_total_points','pace']]
    team1_stats = team1_stats.values.reshape(1, -1)
    team2_stats = team2_stats.values.reshape(1, -1)
    team1_win_percentage = model.predict(team1_stats)
    team2_win_percentage = model.predict(team2_stats)
    if team1_win_percentage > team2_win_percentage:
        print(team1, 'has a', team1_win_percentage[0], 'chance of winning')
    else:
        print(team2, 'has a', team2_win_percentage[0], 'chance of winning')