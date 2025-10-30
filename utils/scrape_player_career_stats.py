import pandas as pd
from nba_api.stats.endpoints.playercareerstats import PlayerCareerStats
import time

def get_player_career_stats(player_id, per_mode36="PerGame", league_id_nullable="00", max_retries=3):
    """
    Fetches NBA player career stats and returns as a dictionary of DataFrames.
    Includes retry logic and detailed error handling.

    Args:
        player_id (str or int): NBA Player ID
        per_mode36 (str, optional): 'PerGame', 'Totals', etc. Defaults to 'PerGame'.
        league_id_nullable (str, optional): League ID. Defaults to '00' (NBA).
        max_retries (int): Number of retry attempts. Defaults to 3.
    
    Returns:
        dict: Keys are dataset names, values are pandas DataFrames
    """
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Fetching career stats for player {player_id}...")
            
            # Create the API endpoint
            pcs = PlayerCareerStats(
                player_id=str(player_id),
                per_mode36=per_mode36,
                league_id_nullable=league_id_nullable
            )
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.6)
            
            # Manual construction with proper error handling
            def create_dataframe(result_set):
                """Helper to safely create DataFrame from result set"""
                try:
                    # Try get_data_frame() method first
                    if hasattr(result_set, 'get_data_frame'):
                        df = result_set.get_data_frame()
                        if df is not None and not df.empty:
                            return df
                    
                    # Fallback to manual construction
                    if hasattr(result_set, 'data') and hasattr(result_set, 'headers'):
                        if result_set.data and result_set.headers:
                            return pd.DataFrame(result_set.data, columns=result_set.headers)
                    
                    return pd.DataFrame()
                except Exception as e:
                    print(f"Error creating dataframe: {e}")
                    return pd.DataFrame()
            
            datasets = {
                "CareerTotalsAllStarSeason": create_dataframe(pcs.career_totals_all_star_season),
                "CareerTotalsCollegeSeason": create_dataframe(pcs.career_totals_college_season),
                "CareerTotalsPostSeason": create_dataframe(pcs.career_totals_post_season),
                "CareerTotalsRegularSeason": create_dataframe(pcs.career_totals_regular_season),
                "SeasonRankingsPostSeason": create_dataframe(pcs.season_rankings_post_season),
                "SeasonRankingsRegularSeason": create_dataframe(pcs.season_rankings_regular_season),
                "SeasonTotalsAllStarSeason": create_dataframe(pcs.season_totals_all_star_season),
                "SeasonTotalsCollegeSeason": create_dataframe(pcs.season_totals_college_season),
                "SeasonTotalsPostSeason": create_dataframe(pcs.season_totals_post_season),
                "SeasonTotalsRegularSeason": create_dataframe(pcs.season_totals_regular_season),
            }
            
            # Check if we got any data
            non_empty = sum(1 for df in datasets.values() if not df.empty)
            if non_empty > 0:
                print(f"✓ Successfully retrieved {non_empty} non-empty datasets")
                return datasets
            else:
                print("⚠ All datasets are empty")
                
        except Exception as e:
            print(f"✗ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
    
    return {}


def get_player_career_stats_simple(player_id):
    """
    Simplified version that just gets the most important data.
    Use this if the full version is having issues.
    """
    try:
        print(f"Fetching career stats for player {player_id}...")
        
        # Get per game stats
        pcs_per_game = PlayerCareerStats(
            player_id=str(player_id),
            per_mode36="PerGame"
        )
        time.sleep(0.6)
        
        # Get totals stats
        pcs_totals = PlayerCareerStats(
            player_id=str(player_id),
            per_mode36="Totals"
        )
        time.sleep(0.6)
        
        # Extract just the regular season data
        per_game_df = pcs_per_game.season_totals_regular_season.get_data_frame()
        totals_df = pcs_totals.season_totals_regular_season.get_data_frame()
        
        print(f"✓ Per Game: {len(per_game_df)} seasons")
        print(f"✓ Totals: {len(totals_df)} seasons")
        
        return {
            "SeasonTotalsRegularSeason_PerGame": per_game_df,
            "SeasonTotalsRegularSeason_Totals": totals_df
        }
        
    except Exception as e:
        print(f"✗ Failed to fetch career stats: {e}")
        return {}
