# Security Policy

## Reporting Security Vulnerabilities

ActuaFlow takes security seriously. If you discover a security vulnerability in ActuaFlow, please **report it privately** rather than using the public issue tracker.

### How to Report

Please email security concerns to:

**Email:** michael@watsondataandrisksolutions.com  
**Subject Line:** `[SECURITY] ActuaFlow Vulnerability Report`

### What to Include

When reporting a security vulnerability, please provide:

1. **Vulnerability Description**
   - Clear explanation of the security issue
   - Type of vulnerability (e.g., injection, buffer overflow, authentication bypass)
   - CVSS score estimate if possible

2. **Affected Component**
   - Which module(s) are affected
   - Which version(s) are vulnerable
   - Python versions affected

3. **Reproduction Steps**
   - Step-by-step instructions to reproduce
   - Minimal code example
   - Expected vs. actual behavior

4. **Impact Assessment**
   - Severity level (critical, high, medium, low)
   - What could an attacker do?
   - Data at risk?

5. **Suggested Fix** (optional)
   - If you have a proposed solution
   - Patch file or code suggestions

### Response Timeline

We will:

1. **Acknowledge receipt** within 48 hours
2. **Assess severity** within 5 business days
3. **Develop fix** within 14 days (depending on complexity)
4. **Release patch** as soon as fix is ready
5. **Credit researcher** (optional) in security advisory

---

## Supported Versions

Security updates are provided for:

| Version | Supported | End of Life |
|---------|-----------|-------------|
| 0.1.0 | ✅ Yes | TBD |
| < 0.1.0 | ❌ No | N/A |

**Note:** As we're pre-1.0, each release is considered a separate major version. Always update to the latest version for security patches.

---

## Known Security Considerations

### Data Handling

**Good News:** ActuaFlow does NOT:
- ❌ Collect user data
- ❌ Transmit data to remote servers
- ❌ Store credentials or sensitive information
- ❌ Include telemetry or analytics
- ❌ Make automatic network requests

**All processing is local** on your machine.

### Dependencies

We monitor security advisories for dependencies:
- numpy
- pandas
- scipy
- statsmodels
- scikit-learn
- polars

We recommend:
- Keeping dependencies updated
- Using dependency scanning tools (e.g., `pip-audit`, Dependabot)
- Reviewing release notes for security patches

### Code Quality

ActuaFlow includes:
- ✅ Type hints (catch many runtime errors)
- ✅ Input validation
- ✅ Error handling
- ✅ Unit & integration tests
- ✅ Exception hierarchy for safe error handling

---

## Security Best Practices

### For Users

1. **Keep Updated**
   ```bash
   pip install --upgrade actuaflow
   ```

2. **Validate Inputs**
   - Always validate data before using with ActuaFlow
   - Check data sources for integrity

3. **Secure Your Data**
   - ActuaFlow doesn't encrypt data
   - Secure sensitive input/output files yourself
   - Use proper access controls on data files

4. **Dependency Updates**
   ```bash
   pip list --outdated
   pip install --upgrade -r requirements.txt
   ```

5. **Report Issues**
   - Email michael@watsondataandrisksolutions.com for security issues

### For Developers

1. **Code Quality**
   - Security-conscious code practices
   - Secure by design principles
   - Regular security updates

2. **Dependencies**
   - Avoid adding dependencies if possible
   - Review security of all dependencies
   - Update dependencies regularly for patches

3. **Testing**
   - Include tests for edge cases
   - Test error handling paths
   - Validate input handling
   - Security test coverage

4. **Documentation**
   - Document security assumptions
   - Note any known limitations
   - Include usage warnings if needed

---

## Security Audit

ActuaFlow is:
- ✅ Open source (code transparency)
- ✅ Public repository (third-party review possible)
- ✅ MPL-2.0 licensed (legal clarity)
- ✅ Professionally maintained
- ⚠️ Not formally audited (budget permitting, this could change)

For critical applications, consider:
- Code review by security professionals
- Dependency audit
- Static analysis tools (bandit, semgrep)
- Dynamic testing in isolated environment

---

## Regulatory & Compliance

### Insurance Industry

If using ActuaFlow for insurance pricing:

1. **Model Governance**
   - Document model assumptions
   - Validate model outputs
   - Have actuaries review models
   - Maintain audit trail

2. **Regulatory Compliance**
   - Know your jurisdiction's requirements
   - Comply with state insurance departments
   - Follow NAIC guidelines if applicable
   - Document regulatory compliance

3. **Data Security**
   - Secure policyholder data
   - Comply with data privacy laws (GDPR, CCPA, etc.)
   - Encryption for transmission
   - Access controls for storage

4. **Professional Standards**
   - Follow ASOPs (Actuarial Standards of Practice)
   - Have EA/FSA review
   - Document methodology
   - Regular model validation

### Data Privacy

ActuaFlow handles no personal data itself, but when used with data:
- ✅ GDPR compliant (no data collection)
- ✅ CCPA compliant (no data collection)
- ✅ HIPAA ready (you control encryption)
- ✅ PBSA ready (you control access)

**You** are responsible for:
- Protecting your data files
- Encrypting data at rest
- Securing data in transit
- Controlling access to results

---

## Third-Party Dependencies Security

All dependencies are:
- ✅ Open source
- ✅ Widely used
- ✅ Actively maintained
- ✅ Security-focused communities

**Dependency Security Scan:**
```bash
pip install pip-audit
pip-audit
```

---

## Incident Response

If a security vulnerability is discovered in a released version:

1. **Patch Development** - Fix is developed and tested
2. **Security Advisory** - CVE requested if applicable
3. **Release Coordination** - Patch released ASAP
4. **Notification** - Users notified via:
   - GitHub security advisory
   - Release notes
   - Direct email (if applicable)

---

## Contact

**For Security Issues:**
- 🔒 Email: michael@watsondataandrisksolutions.com
- 📧 Subject: `[SECURITY] ActuaFlow Vulnerability Report`
- ⏱️ Response: Within 48 hours

**For General Support:**
- 🐛 GitHub Issues: https://github.com/actuaflow/actuaflow/issues
- 💬 GitHub Discussions: https://github.com/actuaflow/actuaflow/discussions

**For Licensing/Legal:**
- See [LICENSING.md](LICENSING.md)
- See [LICENSE](LICENSE)

---

## Acknowledgments

We appreciate the security research community and responsible disclosure practices.

**Vulnerability Reporters:**
(Will be updated as vulnerabilities are responsibly reported and fixed)

---

## Additional Resources

- **OWASP Top 10** - Common vulnerabilities: https://owasp.org/www-project-top-ten/
- **CWE** - Common Weakness Enumeration: https://cwe.mitre.org/
- **CVE Details** - Vulnerability database: https://www.cvedetails.com/
- **Bandit** - Python security linter: https://bandit.readthedocs.io/
- **pip-audit** - Dependency vulnerability scanner: https://github.com/pypa/pip-audit

---

**Last Updated:** 2026  
**License:** Mozilla Public License v2.0  
**Author:** Michael Watson
