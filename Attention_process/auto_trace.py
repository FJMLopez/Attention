import logging
import time
import inspect
import sys
from functools import wraps

def traced(func):
    """Décorateur qui loggue les appels, les valeurs, les exceptions et les durées."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("TRACE")
        logger.debug(f"→ Appel de {func.__name__} args={args}, kwargs={kwargs}")
        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000
            logger.debug(f"← Retour de {func.__name__}: {result} ({duration:.2f} ms)")
            return result

        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            logger.exception(
                f"⚠️ Exception dans {func.__name__} après {duration:.2f} ms : {e}"
            )
            raise

    return wrapper


def apply_tracing(module_globals):
    """
    Applique automatiquement `@traced` à toutes les fonctions d'un fichier.
    À appeler en haut du fichier à tracer :
        apply_tracing(globals())
    """
    for name, obj in list(module_globals.items()):
        if inspect.isfunction(obj):
            module_globals[name] = traced(obj)
        elif inspect.isclass(obj):
            # Pour les méthodes des classes
            for attr_name, attr_val in obj.__dict__.items():
                if inspect.isfunction(attr_val):
                    setattr(obj, attr_name, traced(attr_val))
