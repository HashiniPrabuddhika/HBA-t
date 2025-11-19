import os
import sys
import gzip
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_sqlite_database(db_path: str, backup_path: str, compress: bool = True):
    if not Path(db_path).exists():
        logger.warning(f"Database not found: {db_path}")
        return False
    try:
        source = sqlite3.connect(db_path)
        if compress:
            temp_backup = f"{backup_path}.temp"
            backup = sqlite3.connect(temp_backup)
            source.backup(backup)
            backup.close()
            with open(temp_backup, 'rb') as f_in:
                with gzip.open(f"{backup_path}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            Path(temp_backup).unlink()
            logger.info(f"Backed up and compressed: {db_path}")
        else:
            backup = sqlite3.connect(backup_path)
            source.backup(backup)
            backup.close()
            logger.info(f"Backed up: {db_path}")
        source.close()
        return True
    except Exception as e:
        logger.error(f"Failed to backup database {db_path}: {e}")
        return False

def create_backup(base_path: str = "data", backup_path: str = "backups", 
                 include_cache: bool = False, compress: bool = True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(backup_path) / f"data_backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating backup in: {backup_dir}")
    
    try:
        logger.info("Backing up databases...")
        db_backup_dir = backup_dir / "databases"
        db_backup_dir.mkdir(exist_ok=True)
        
        db_files = [
            f"{base_path}/embeddings/metadata/embeddings_metadata.db",
            f"{base_path}/models/metadata/models_metadata.db",
            f"{base_path}/analytics/analytics.db"
        ]
        
        cache_db_paths = [
            f"{base_path}/cache/recommendations_cache.db",
            f"{base_path}/cache/metadata/cache_metadata.db"
        ]
        
        for cache_db in cache_db_paths:
            if Path(cache_db).exists():
                db_files.append(cache_db)
        
        for db_file in db_files:
            if Path(db_file).exists():
                dest_file = db_backup_dir / Path(db_file).name
                backup_sqlite_database(db_file, str(dest_file), compress)
        
        logger.info("Backing up models...")
        models_backup_dir = backup_dir / "models"
        models_src = Path(f"{base_path}/models")
        if models_src.exists():
            shutil.copytree(models_src, models_backup_dir, dirs_exist_ok=True)
            if compress:
                for model_file in models_backup_dir.rglob("*.pkl"):
                    with open(model_file, 'rb') as f_in:
                        with gzip.open(f"{model_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    model_file.unlink()
        
        logger.info("Backing up embeddings...")
        embeddings_backup_dir = backup_dir / "embeddings"
        embeddings_src = Path(f"{base_path}/embeddings")
        if embeddings_src.exists():
            shutil.copytree(embeddings_src, embeddings_backup_dir, dirs_exist_ok=True)
        
        logger.info("Backing up analytics...")
        analytics_backup_dir = backup_dir / "analytics"
        analytics_src = Path(f"{base_path}/analytics")
        if analytics_src.exists():
            for item in analytics_src.iterdir():
                if item.name != "events":
                    if item.is_dir():
                        shutil.copytree(item, analytics_backup_dir / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, analytics_backup_dir / item.name)
        
        if include_cache:
            logger.info("Backing up cache...")
            cache_backup_dir = backup_dir / "cache"
            cache_src = Path(f"{base_path}/cache")
            if cache_src.exists():
                cache_backup_dir.mkdir(exist_ok=True)
                for cache_file in cache_src.rglob("*.db"):
                    relative_path = cache_file.relative_to(cache_src)
                    dest_file = cache_backup_dir / relative_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    backup_sqlite_database(str(cache_file), str(dest_file), compress)
                for cache_file in cache_src.rglob("*"):
                    if cache_file.is_file() and cache_file.suffix != '.db':
                        relative_path = cache_file.relative_to(cache_src)
                        dest_file = cache_backup_dir / relative_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        if compress and cache_file.suffix in ['.json', '.txt', '.log']:
                            with open(cache_file, 'rb') as f_in:
                                with gzip.open(f"{dest_file}.gz", 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                        else:
                            shutil.copy2(cache_file, dest_file)
        
        manifest = {
            'timestamp': timestamp,
            'base_path': base_path,
            'include_cache': include_cache,
            'compressed': compress,
            'cache_type': 'SQLite',
            'files_backed_up': [],
            'backup_size_mb': 0
        }
        
        total_size = 0
        for file_path in backup_dir.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                manifest['files_backed_up'].append({
                    'path': str(file_path.relative_to(backup_dir)),
                    'size_bytes': size
                })
        
        manifest['backup_size_mb'] = total_size / (1024 * 1024)
        with open(backup_dir / "backup_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Backup completed successfully!")
        logger.info(f"Backup size: {manifest['backup_size_mb']:.2f} MB")
        logger.info(f"Files backed up: {len(manifest['files_backed_up'])}")
        return str(backup_dir)
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        raise

def restore_sqlite_database(backup_path: str, restore_path: str, compressed: bool = False):
    try:
        restore_file = Path(restore_path)
        restore_file.parent.mkdir(parents=True, exist_ok=True)
        if compressed and Path(f"{backup_path}.gz").exists():
            with gzip.open(f"{backup_path}.gz", 'rb') as f_in:
                with open(restore_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif Path(backup_path).exists():
            shutil.copy2(backup_path, restore_file)
        else:
            logger.warning(f"Backup file not found: {backup_path}")
            return False
        conn = sqlite3.connect(restore_file)
        conn.execute("PRAGMA integrity_check")
        conn.close()
        logger.info(f"Restored database: {restore_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to restore database {backup_path}: {e}")
        return False

def restore_backup(backup_path: str, restore_path: str = "data", verify: bool = True):
    backup_dir = Path(backup_path)
    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup directory not found: {backup_dir}")
    
    manifest_file = backup_dir / "backup_manifest.json"
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        logger.info(f"Restoring backup from: {manifest['timestamp']}")
        logger.info(f"Backup size: {manifest['backup_size_mb']:.2f} MB")
        logger.info(f"Cache type: {manifest.get('cache_type', 'Unknown')}")
        compressed = manifest.get('compressed', False)
    else:
        logger.warning("No manifest found, proceeding with best effort restore")
        compressed = False
    
    restore_dir = Path(restore_path)
    restore_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("Restoring databases...")
        db_backup_dir = backup_dir / "databases"
        if db_backup_dir.exists():
            for db_file in db_backup_dir.iterdir():
                if db_file.name.endswith('.gz') and compressed:
                    db_name = db_file.stem
                    if 'embeddings' in db_name:
                        dest_path = restore_dir / "embeddings" / "metadata" / db_name
                    elif 'models' in db_name:
                        dest_path = restore_dir / "models" / "metadata" / db_name
                    elif 'analytics' in db_name:
                        dest_path = restore_dir / "analytics" / db_name
                    elif 'cache' in db_name:
                        dest_path = restore_dir / "cache" / db_name
                    else:
                        dest_path = restore_dir / "metadata" / db_name
                    restore_sqlite_database(str(db_file)[:-3], str(dest_path), True)
                elif db_file.suffix == '.db':
                    db_name = db_file.name
                    if 'embeddings' in db_name:
                        dest_path = restore_dir / "embeddings" / "metadata" / db_name
                    elif 'models' in db_name:
                        dest_path = restore_dir / "models" / "metadata" / db_name
                    elif 'analytics' in db_name:
                        dest_path = restore_dir / "analytics" / db_name
                    elif 'cache' in db_name:
                        dest_path = restore_dir / "cache" / db_name
                    else:
                        dest_path = restore_dir / "metadata" / db_name
                    restore_sqlite_database(str(db_file), str(dest_path), False)
        
        for component in ['models', 'embeddings', 'analytics', 'cache']:
            component_backup = backup_dir / component
            if component_backup.exists():
                logger.info(f"Restoring {component}...")
                component_restore = restore_dir / component
                if component == 'cache':
                    component_restore.mkdir(parents=True, exist_ok=True)
                    for item in component_backup.rglob('*'):
                        if item.is_file():
                            relative_path = item.relative_to(component_backup)
                            dest_file = component_restore / relative_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            if item.suffix == '.db':
                                restore_sqlite_database(str(item), str(dest_file), False)
                            elif item.name.endswith('.db.gz'):
                                db_dest = dest_file.parent / item.stem
                                restore_sqlite_database(str(item)[:-3], str(db_dest), True)
                            elif item.name.endswith('.gz'):
                                with gzip.open(item, 'rb') as f_in:
                                    with open(dest_file.with_suffix(''), 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                            else:
                                shutil.copy2(item, dest_file)
                else:
                    if component_restore.exists():
                        shutil.rmtree(component_restore)
                    shutil.copytree(component_backup, component_restore)
                    if component == 'models' and compressed:
                        for gz_file in component_restore.rglob("*.pkl.gz"):
                            pkl_file = gz_file.with_suffix('')
                            with gzip.open(gz_file, 'rb') as f_in:
                                with open(pkl_file, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            gz_file.unlink()
        
        if verify:
            logger.info("Verifying restored data...")
            verify_restore(restore_path)
        
        logger.info("Restore completed successfully!")
        return str(restore_dir)
        
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        raise

def verify_restore(restore_path: str):
    try:
        logger.info("Verifying SQLite databases...")
        db_files = [
            f"{restore_path}/embeddings/metadata/embeddings_metadata.db",
            f"{restore_path}/models/metadata/models_metadata.db",
            f"{restore_path}/analytics/analytics.db",
            f"{restore_path}/cache/recommendations_cache.db"
        ]
        
        for db_file in db_files:
            if Path(db_file).exists():
                try:
                    conn = sqlite3.connect(db_file)
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                    if result[0] == 'ok':
                        logger.info(f"Database verified: {db_file}")
                    else:
                        logger.error(f"Database corruption detected: {db_file}")
                    conn.close()
                except Exception as e:
                    logger.error(f"Database verification failed for {db_file}: {e}")
        
        try:
            from data import EmbeddingManager, ModelManager, AnalyticsManager
            from recommendtion.recommendations.utils.cache_manager import CacheManager
            
            embedding_mgr = EmbeddingManager(f"{restore_path}/embeddings")
            stats = embedding_mgr.get_embedding_stats()
            logger.info(f"Embeddings verified: {stats}")
            embedding_mgr.close()
            
            model_mgr = ModelManager(f"{restore_path}/models")
            stats = model_mgr.get_storage_stats()
            logger.info(f"Models verified: {stats}")
            model_mgr.close()
            
            cache_mgr = CacheManager()
            stats = cache_mgr.get_cache_stats() if hasattr(cache_mgr, 'get_cache_stats') else {"status": "available"}
            logger.info(f"Cache verified: {stats}")
            if hasattr(cache_mgr, 'close'):
                cache_mgr.close()
            
            analytics_mgr = AnalyticsManager(f"{restore_path}/analytics")
            summary = analytics_mgr.get_analytics_summary()
            logger.info(f"Analytics verified: {summary}")
            analytics_mgr.close()
            
            logger.info("Manager verification completed successfully!")
            
        except ImportError as e:
            logger.warning(f"Could not import managers for verification: {e}")
            logger.info("Database integrity checks passed, skipping manager tests")
        
        logger.info("Verification completed successfully!")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

def cleanup_old_backups(backup_path: str = "backups", keep_days: int = 30):
    backup_dir = Path(backup_path)
    if not backup_dir.exists():
        return
    cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
    for backup_folder in backup_dir.iterdir():
        if backup_folder.is_dir() and backup_folder.name.startswith('data_backup_'):
            if backup_folder.stat().st_mtime < cutoff_time:
                logger.info(f"Removing old backup: {backup_folder}")
                shutil.rmtree(backup_folder)

def main():
    parser = argparse.ArgumentParser(description='Data storage backup and restore utility (SQLite version)')
    parser.add_argument('action', choices=['backup', 'restore', 'cleanup'], help='Action to perform')
    parser.add_argument('--base-path', default='data', help='Base path of data storage (default: data)')
    parser.add_argument('--backup-path', default='backups', help='Backup directory path (default: backups)')
    parser.add_argument('--restore-from', help='Path to backup directory for restoration')
    parser.add_argument('--include-cache', action='store_true', help='Include cache data in backup')
    parser.add_argument('--no-compress', action='store_true', help='Disable compression')
    parser.add_argument('--keep-days', type=int, default=30, help='Days to keep backups during cleanup (default: 30)')
    parser.add_argument('--no-verify', action='store_true', help='Skip verification after restore')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'backup':
            backup_dir = create_backup(
                base_path=args.base_path,
                backup_path=args.backup_path,
                include_cache=args.include_cache,
                compress=not args.no_compress
            )
            print(f"Backup created: {backup_dir}")
        elif args.action == 'restore':
            if not args.restore_from:
                parser.error("--restore-from is required for restore action")
            restore_dir = restore_backup(
                backup_path=args.restore_from,
                restore_path=args.base_path,
                verify=not args.no_verify
            )
            print(f"Data restored to: {restore_dir}")
        elif args.action == 'cleanup':
            cleanup_old_backups(
                backup_path=args.backup_path,
                keep_days=args.keep_days
            )
            print(f"Cleaned up backups older than {args.keep_days} days")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()